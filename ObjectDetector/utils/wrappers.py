import numpy as np

############################################
#              Boxmot Wrapper              #
############################################
def supervision_to_boxmot(detections):
    """
    Converts detections from Ultralytics supervision format to (x1, y1, x2, y2, confidence, class) format.

    Args:
        detections: Detections in Ultralytics supervision format

    Returns:
        np.ndarray: Array of detections in the desired format N x (x1, y1, x2, y2, confidence, class)
    """
    xyxy = detections.xyxy
    confidence = detections.confidence
    class_id = detections.class_id

    # Create an empty array to store converted detections
    num_detections = len(xyxy)
    converted_dets = np.zeros((num_detections, 6))

    for i in range(num_detections):
        x1, y1, x2, y2 = xyxy[i]
        conf = confidence[i]
        cls = class_id[i]

        converted_dets[i] = [x1, y1, x2, y2, conf, cls]
    
    return converted_dets

def boxmot_to_supervision(boxmot_detections, supervision_object, object_detector_model):
    """
    Converts detections from BoxMot format to Ultralytics supervision format.

    Args:
        boxmot_detections (np.ndarray): Detections in BoxMot format (N x 6).
        supervision_object: An existing Ultralytics supervision object (e.g., Detections).

    Returns:
        dict: Updated supervision object with converted detections.
    """
    if boxmot_detections.size == 0:
        return supervision_object
    xyxy = boxmot_detections[:, :4]  # Bounding box coordinates
    track_id = boxmot_detections[:, 4]  # Tracker ID
    confidence = boxmot_detections[:, 5] # Confidence scores
    class_id = boxmot_detections[:, 6]  # Class IDs

    xyxy_float32 = np.array([row.astype(np.float32) for row in xyxy])

    class_id_int32 = class_id.astype(np.int32)
    track_id_int32 = track_id.astype(np.int32)

    class_names = np.array([object_detector_model.names.get(cid) for cid in class_id_int32])

    # Update the existing supervision object
    supervision_object.xyxy = xyxy_float32
    supervision_object.confidence = confidence
    supervision_object.class_id = class_id_int32
    supervision_object.tracker_id = track_id_int32
    supervision_object.data["class_name"] = class_names

    return supervision_object