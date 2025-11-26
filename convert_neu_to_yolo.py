from pathlib import Path
import xml.etree.ElementTree as ET
import shutil


CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def parse_voc_xml(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        objects.append({
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "width": width,
            "height": height
        })
    return objects


def voc_to_yolo(obj: dict) -> str:
    class_id = CLASSES.index(obj["name"])
    x_center = ((obj["xmin"] + obj["xmax"]) / 2) / obj["width"]
    y_center = ((obj["ymin"] + obj["ymax"]) / 2) / obj["height"]
    box_width = (obj["xmax"] - obj["xmin"]) / obj["width"]
    box_height = (obj["ymax"] - obj["ymin"]) / obj["height"]
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def convert_split(src_dir: Path, dst_dir: Path, split: str) -> int:
    img_src = src_dir / split / "images"
    ann_src = src_dir / split / "annotations"

    img_dst = dst_dir / "images" / split
    lbl_dst = dst_dir / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for class_dir in img_src.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.jpg"):
            xml_name = img_path.stem + ".xml"
            xml_path = ann_src / xml_name

            if not xml_path.exists():
                continue

            shutil.copy(img_path, img_dst / img_path.name)

            objects = parse_voc_xml(xml_path)
            yolo_lines = [voc_to_yolo(obj) for obj in objects]

            txt_path = lbl_dst / (img_path.stem + ".txt")
            txt_path.write_text("\n".join(yolo_lines))
            count += 1

    return count


def create_yaml(dst_dir: Path) -> None:
    yaml_content = f"""path: {dst_dir.absolute()}
train: images/train
val: images/validation

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""
    (dst_dir / "neu-det.yaml").write_text(yaml_content)


def main():
    src_dir = Path("data/neu-det/NEU-DET")
    dst_dir = Path("data/neu-det-yolo")

    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    train_count = convert_split(src_dir, dst_dir, "train")
    val_count = convert_split(src_dir, dst_dir, "validation")

    create_yaml(dst_dir)

    print(f"Converted {train_count} training images")
    print(f"Converted {val_count} validation images")
    print(f"Dataset saved to: {dst_dir.absolute()}")


if __name__ == "__main__":
    main()
