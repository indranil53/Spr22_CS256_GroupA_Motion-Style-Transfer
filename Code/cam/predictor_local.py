from scipy.spatial import ConvexHull
import torch
import yaml
from modules.keypoint_detector import KPDetector
from modules.keypoint_detector import HEEstimator
from modules.generator import OcclusionAwareSPADEGenerator
#from modules.generator import OcclusionAwareGenerator
from sync_batchnorm import DataParallelWithCallback
import numpy as np
import face_alignment
import torch.nn.functional as F


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().detach().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().detach().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    # if use_relative_movement:
    #     kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
    #     kp_value_diff *= adapt_movement_scale
    #     kp_new['value'] = kp_value_diff + kp_source['value']

    #     if use_relative_jacobian:
    #         jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
    #         kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, estimate_jacobian=False, free_view=False, yaw=0, pitch=0, roll=0):
    kp = kp_canonical['value']
    if not free_view:
        yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)
    else:
        if yaw is not None:
            yaw = torch.tensor([yaw]).cuda()
        else:
            yaw = he['yaw']
            yaw = headpose_pred_to_degree(yaw)
        if pitch is not None:
            pitch = torch.tensor([pitch]).cuda()
        else:
            pitch = he['pitch']
            pitch = headpose_pred_to_degree(pitch)
        if roll is not None:
            roll = torch.tensor([roll]).cuda()
        else:
            roll = he['roll']
            roll = headpose_pred_to_degree(roll)

    t, exp = he['t'], he['exp']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}

def to_tensor(a):
    return torch.tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255


class PredictorLocal:
    def __init__(self, config_path, checkpoint_path, relative=False, adapt_movement_scale=False, device=None, enc_downscale=1):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.relative = relative
        self.adapt_movement_scale = adapt_movement_scale
        self.start_frame = None
        self.start_frame_kp = None
        self.kp_driving_initial = None
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.generator, self.kp_detector,self.he_estimator = self.load_checkpoints()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device=self.device)
        self.source = None
        self.kp_source = None
        self.enc_downscale = enc_downscale

    def load_checkpoints(self,cpu=False):
        with open(self.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        if not cpu:
            generator.cuda()
        generator.to(self.device)
    
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        if not cpu:
            kp_detector.cuda()
        kp_detector.to(self.device)

        he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
        if not cpu:
            he_estimator.cuda()
        he_estimator.to(self.device)
    
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        he_estimator.load_state_dict(checkpoint['he_estimator'])
    
        generator.eval()
        kp_detector.eval()
        he_estimator.eval()
        
        return generator, kp_detector, he_estimator

    def reset_frames(self):
        self.kp_driving_initial = None

    def set_source_image(self, source_image):
        self.source = to_tensor(source_image).to(self.device)
        self.kp_canonical = self.kp_detector(self.source)
        self.he_source = self.he_estimator(self.source)
        self.kp_source = keypoint_transformation(self.kp_canonical, self.he_source)

        if self.enc_downscale > 1:
            h, w = int(self.source.shape[2] / self.enc_downscale), int(self.source.shape[3] / self.enc_downscale)
            source_enc = torch.nn.functional.interpolate(self.source, size=(h, w), mode='bilinear')
        else:
            source_enc = self.source

        self.generator.encode_source(source_enc)

    def predict(self, driving_frame):
        assert self.kp_source is not None, "call set_source_image()"
        
        with torch.no_grad():
            driving = to_tensor(driving_frame).to(self.device)

            if self.kp_driving_initial is None:
                self.kp_driving_initial = self.kp_detector(driving)
                self.start_frame = driving_frame.copy()
                self.start_frame_kp = self.get_frame_kp(driving_frame)

            #kp_driving = self.kp_detector(driving)
            self.he_driving = self.he_estimator(driving)
            kp_driving= keypoint_transformation(self.kp_canonical,self.he_driving, free_view=True, yaw=20, pitch=0, roll=0)
            kp_norm = normalize_kp(kp_source=self.kp_source, kp_driving=kp_driving,
                                kp_driving_initial=self.kp_driving_initial, use_relative_movement=self.relative,
                                use_relative_jacobian=False, adapt_movement_scale=self.adapt_movement_scale)

            out = self.generator(self.source, kp_source=self.kp_source, kp_driving=kp_norm)

            out = np.transpose(out['prediction'].data.cpu().detach().numpy(), [0, 2, 3, 1])[0]
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

            return out

    def get_frame_kp(self, image):
        kp_landmarks = self.fa.get_landmarks(image)
        if kp_landmarks:
            kp_image = kp_landmarks[0]
            kp_image = self.normalize_alignment_kp(kp_image)
            return kp_image
        else:
            return None

    @staticmethod
    def normalize_alignment_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp
    
    def get_start_frame(self):
        return self.start_frame

    def get_start_frame_kp(self):
        return self.start_frame_kp
