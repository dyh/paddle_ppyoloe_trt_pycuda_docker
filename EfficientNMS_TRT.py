import argparse
import onnx
import onnx_graphsurgeon as gs
import onnxsim
import numpy as np
from onnx import shape_inference
from collections import OrderedDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser('nms tool')
    parser.add_argument('--weights_type', type=str, help='model weights type')
    parser.add_argument('--input_path', type=str, help='input onnx')
    parser.add_argument('--save_path', type=str, help='output onnx')
    parser.add_argument('--class_num', type=int, default=80, help='class number')
    parser.add_argument('--score_threshold', type=float, default=0.25, help='score threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='iou threshold')

    args = parser.parse_args()

    print('args:', args)

    WEIGHTS_TYPE = args.weights_type
    INPUT_PATH = args.input_path
    SAVE_PATH = args.save_path
    CLASS_NUM = args.class_num
    SCORE_THRESHOLD = args.score_threshold
    IOU_THRESHOLD = args.iou_threshold

    if (WEIGHTS_TYPE == "s"):
        mul_name = 'Mul_78'
    elif (WEIGHTS_TYPE == "m"):
        mul_name = 'Mul_100'
    elif (WEIGHTS_TYPE == "l"):
        mul_name = 'Mul_122'
    elif (WEIGHTS_TYPE == "x"):
        mul_name = 'Mul_144'

    gs_graph = gs.import_onnx(onnx.load(INPUT_PATH))
    # fold constants
    gs_graph.fold_constants()
    gs_graph.cleanup().toposort()

    Mul = [node for node in gs_graph.nodes if node.name == mul_name][0]
    Concat_14 = [node for node in gs_graph.nodes if node.name == 'Concat_14'][0]

    scores = gs.Variable(name='scores', shape=[1, 8400, CLASS_NUM], dtype=np.float32)
    Transpose = gs.Node(name='lastTranspose', op='Transpose',
                        inputs=[Concat_14.outputs[0]],
                        outputs=[scores],
                        attrs=OrderedDict(perm=[0, 2, 1]))
    gs_graph.nodes.append(Transpose)

    Mul.outputs[0].name = 'boxes'
    gs_graph.inputs = [gs_graph.inputs[0]]
    gs_graph.outputs = [Mul.outputs[0], scores]
    gs_graph.outputs[0].dtype = np.float32
    gs_graph.outputs[1].dtype = np.float32

    gs_graph.cleanup().toposort()
    onnx_graph = shape_inference.infer_shapes(gs.export_onnx(gs_graph))
    onnx_graph, check = onnxsim.simplify(onnx_graph)

    gs_graph = gs.import_onnx(onnx_graph)
    op_inputs = gs_graph.outputs
    op = "EfficientNMS_TRT"
    attrs = {
        "plugin_version": "1",
        "background_class": -1,
        "max_output_boxes": 100,
        "score_threshold": SCORE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "score_activation": False,
        "box_coding": 0,
    }

    output_num_detections = gs.Variable(
        name="num_dets",
        dtype=np.int32,
        shape=[1, 1],
    )
    output_boxes = gs.Variable(
        name="det_boxes",
        dtype=np.float32,
        shape=[1, 100, 4],
    )
    output_scores = gs.Variable(
        name="det_scores",
        dtype=np.float32,
        shape=[1, 100],
    )
    output_labels = gs.Variable(
        name="det_classes",
        dtype=np.int32,
        shape=[1, 100],
    )
    op_outputs = [
        output_num_detections, output_boxes, output_scores, output_labels
    ]

    TRT = gs.Node(op=op, name="batched_nms", inputs=op_inputs, outputs=op_outputs, attrs=attrs)
    gs_graph.nodes.append(TRT)
    gs_graph.outputs = op_outputs
    gs_graph.cleanup().toposort()

    onnx.save(gs.export_onnx(gs_graph), SAVE_PATH)

    print("nms finished", SAVE_PATH)
