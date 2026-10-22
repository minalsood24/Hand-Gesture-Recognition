import math


def distance(a, b):
    """Calculate Euclidean distance between two MediaPipe landmarks."""
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def angle_between(a, b, c):
    """
    Calculate angle ABC in degrees.
    B is the vertex.
    """
    ba_x = a.x - b.x
    ba_y = a.y - b.y

    bc_x = c.x - b.x
    bc_y = c.y - b.y

    dot_product = ba_x * bc_x + ba_y * bc_y

    magnitude_ba = math.sqrt(ba_x ** 2 + ba_y ** 2)
    magnitude_bc = math.sqrt(bc_x ** 2 + bc_y ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    # Prevent floating-point errors
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))


def fingers_up_count(landmarks):
    """
    Estimate how many fingers are extended.

    Uses the y-coordinate for the four fingers and
    wrist-distance comparison for the thumb.
    """

    count = 0

    # Thumb
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    wrist = landmarks[0]

    thumb_tip_distance = distance(thumb_tip, wrist)
    thumb_ip_distance = distance(thumb_ip, wrist)

    if thumb_tip_distance > thumb_ip_distance:
        count += 1

    # Index, Middle, Ring, Pinky
    finger_pairs = [
        (8, 6),    # index
        (12, 10),  # middle
        (16, 14),  # ring
        (20, 18)   # pinky
    ]

    for tip, pip in finger_pairs:
        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    return count


def extract_features(landmarks):
    """
    Extract the same eight features used by the ML model.

    Returns:
        list of 8 numerical features
    """

    wrist = landmarks[0]

    # Use wrist -> middle MCP as a scale reference.
    # This makes the system less sensitive to hand distance
    # from the camera.
    scale = distance(wrist, landmarks[9])

    if scale < 0.001:
        scale = 0.001

    thumb_index_dist = distance(landmarks[4], landmarks[8]) / scale

    index_extension = distance(landmarks[8], wrist) / scale
    middle_extension = distance(landmarks[12], wrist) / scale
    ring_extension = distance(landmarks[16], wrist) / scale
    pinky_extension = distance(landmarks[20], wrist) / scale

    palm_width = distance(landmarks[5], landmarks[17]) / scale

    # Angle between wrist -> index MCP and wrist -> thumb tip
    thumb_angle = angle_between(
        landmarks[5],
        wrist,
        landmarks[4]
    )

    fingers_count = fingers_up_count(landmarks)

    return [
        thumb_index_dist,
        index_extension,
        middle_extension,
        ring_extension,
        pinky_extension,
        palm_width,
        thumb_angle,
        fingers_count
    ]


FEATURE_NAMES = [
    "thumb_index_dist",
    "index_extension",
    "middle_extension",
    "ring_extension",
    "pinky_extension",
    "palm_width",
    "thumb_angle",
    "fingers_up_count"
]