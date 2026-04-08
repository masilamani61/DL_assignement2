# Count how many test images have real bboxes
from data.pets_dataset import OxfordIIITPetDataset
import os

ds = OxfordIIITPetDataset("data", split="test")
real_bbox = 0
fake_bbox = 0

for i in range(len(ds)):
    img_name = ds.samples[i][0]
    xml_path = os.path.join("data", "annotations", "xmls", img_name + ".xml")
    if os.path.exists(xml_path):
        real_bbox += 1
    else:
        fake_bbox += 1

print(f"Real bboxes: {real_bbox}")
print(f"Fake bboxes: {fake_bbox}")
print(f"Total: {len(ds)}")