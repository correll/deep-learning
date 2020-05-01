"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor, IntTensor
from sort import *

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.prev_trans_detection = None
        self.count = 0
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.mot_tracker = Sort() 
        self.threshold = 0.8
        self.tracked_objects = None
        self.trans_detection = None
#         self.kalman_filter = KalmanBoxTracker()

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        detector_losses = {}
        proposal_losses = {}
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        print("Run features through the backbone")
        features = self.backbone(images.tensors)
#         print(features)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        
        if self.count % 5 == 0 or self.count < 20:
            print("Run features through the proposals:")
            proposals, proposal_losses = self.rpn(images, features, targets)
#             print("proposal",proposals, proposal_losses)
            print("Run features through the roi:")
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
#             print("format for detector_losses", detector_losses)
#             print("before postprocess",detections)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
#             print("after postprocess",detections)
            pred_class = [i for i in list(detections[0]['labels'].numpy())] # Get the Prediction cls
            pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(detections[0]['boxes'].detach().numpy())] # Bounding boxes
            pred_score = list(detections[0]['scores'].detach().numpy()) # get prediction score 
            pred_t = [pred_score.index(x) for x in pred_score if x > self.threshold][-1] # Get list of index with score greater than threshold.
            pred_boxes = pred_boxes[:pred_t+1]
            pred_class = pred_class[:pred_t+1]
            pred_score = pred_score[:pred_t+1]
            
            trans_detection =[]
            for i, j, k in zip( pred_boxes,pred_score, pred_class):
                trans_detection.append( i + [j, j,k])
            self.trans_detection = Tensor(trans_detection)
            self.tracked_objects = self.mot_tracker.update(self.trans_detection.cpu())
#             print("tracked", self.tracked_objects)
        else:
#           trans_detection = self.prev_trans_detection
#             self.tracked_objects = self.mot_tracker.update(self.tracked_objects,True)
            self.tracked_objects = self.mot_tracker.update(self.trans_detection,True)

#             print("tracked withou prediction", self.tracked_objects)
            
#         print("trans_detection", trans_detection)
#         tracked_objects = self.mot_tracker.update(trans_detection.cpu())
#         print("tracked",tracked_objects)

#             print("post process detection:", detections)
#             print(detections)
#         # Get list of index with score greater than threshold.
#         pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(detections[0]['boxes'].detach().numpy())] # Bounding boxes
#           pred_labels = [[(i[0], i[1]), (i[2], i[3])] for i in list(detections[0 ['labels'].detach().numpy())]
#         pred_score = list(detections[0]['scores'].detach().numpy())
#         pred_t = [pred_score.index(x) for x in pred_score if x > 0.7][-1] 
#         pred_boxes = pred_boxes[:pred_t+1]
#         pred_class = pred_class[:pred_t+1]
        ## labels scores and boxes
#          detections = {"labels":}
#         else:
#             detections = self.prevdetection
            
    
#         self.prev_trans_detection = trans_detection
        self.count += 1
        print(self.count)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        detection_with_tracker = {"labels":[], "boxes":[],"tracker_id":[]}
        for i in self.tracked_objects:
#             print(i[0:4])
            detection_with_tracker["boxes"].append(i[0:4])
            detection_with_tracker["labels"].append(i[5])
            detection_with_tracker["tracker_id"].append(i[4])
        detection_with_tracker["boxes"]=Tensor(detection_with_tracker["boxes"])
        detection_with_tracker["labels"]=IntTensor(detection_with_tracker["labels"])
        detection_with_tracker["tracker_id"]=IntTensor(detection_with_tracker["tracker_id"])
        
#         print(detection_with_tracker)  
          

        if torch.jit.is_scripting():
            warnings.warn("RCNN always returns a (Losses, Detections tuple in scripting)")
            return (losses, detection_with_tracker)
        else:
            return self.eager_outputs(losses, detection_with_tracker)
