"""
Số reader đánh dấu mỗi finding trên mỗi ảnh, đọc từ annotation gốc.

Dùng cho label_policy ignore_minority / soft_votes của classification loss: ca
chỉ có 1/3 reader đánh dấu đang được coi là âm tính theo quy tắc đa số.
Cách đếm giống positive_reader_counts trong src/data/annotations.py (mỗi reader
có ít nhất một box của finding đó thì tính một phiếu).
"""

import pandas as pd


def build_reader_vote_table(annotation_csv, findings, image_ids=None):
    """DataFrame index=image_id, cột=findings, giá trị=số reader đánh dấu (int)."""
    annotations = pd.read_csv(annotation_csv, usecols=["image_id", "class_name", "rad_id"])
    if image_ids is not None:
        annotations = annotations[annotations["image_id"].isin(set(image_ids))]
    annotations = annotations[annotations["class_name"].isin(findings)]
    votes = (
        annotations.groupby(["image_id", "class_name"])["rad_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=list(findings), fill_value=0)
    )
    if image_ids is not None:
        votes = votes.reindex(list(image_ids), fill_value=0)
    return votes.astype(int)
