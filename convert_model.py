from PIL import Image
import torch
import torchvision
import argparse
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # projects root directory

from src.model import get_model_instance_segmentation

def convert_model_to_onnx(path_to_ckpt:str, onnx_pth:str="output/maskrcnn.onnx"):
    raise NotImplementedError # after convertation model outputs empty lists....
    cpu_device = torch.device("cpu")
    model = get_model_instance_segmentation(2, pretrained=False, state_dict_pth=path_to_ckpt)
    model.eval()
    model.to(cpu_device)

    # fh = 'data/images/livecell_train_val_images/A172_Phase_B7_1_00d00h00m_3.tif'
    # image = Image.open(fh, mode='r')
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # img = transform(image)
    # img.unsqueeze_(0)
    input_data = torch.rand((1, 1, 520, 704), device = cpu_device)
    with torch.no_grad():
        out = model(input_data)

    # Export the PyTorch model to ONNX format
    torch.onnx.export(model,
                    input_data,
                    onnx_pth,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    )

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_torch_model', type=str, help='path to model state dict')
    # add model config!
    parser.add_argument('--path_to_onnx_model', type=str, default=ROOT / 'output/maskrcnn.onnx', help='where to put onnx model')

    return parser.parse_args()


def main(opt):
    convert_model_to_onnx(opt.path_to_torch_model, opt.path_to_onnx_model)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)