from lxml.etree import ElementTree

from configs.mask_configs import KAGGLE_MASK_LABELS
from data_utils.pascal import read_pascal


def xml_to_dict(filename) -> dict:
    tree = ElementTree(file=filename)
    root = tree.getroot()

    def parse(xml_root):
        """
        Parse đệ quy
            - Nếu có xml_root không thẻ con, ta sẽ lấy ra giá trị
            - Với mỗi thẻ con ta sẽ gọi đệ quy để lấy giá trị.
                + Nếu xuất hiện một thẻ có tag đã tồn tại trong info thì sẽ chuyển sang list
        :param xml_root:
        :return:
        """
        if len(xml_root) == 0:
            try:
                return float(xml_root.text)
            except ValueError:
                return xml_root.text
        info = {}
        for elm in xml_root:
            value = parse(elm)
            existed = info.get(elm.tag)
            if existed is None:
                info[elm.tag] = value
            elif type(existed) == list:
                info[elm.tag] += value
        if xml_root.tag == 'object':
            return [info]
        return info

    return parse(root)


def read_kaggle_mask(annotation, images_dir, target: dict):
    return read_pascal(annotation, images_dir, KAGGLE_MASK_LABELS, target)
