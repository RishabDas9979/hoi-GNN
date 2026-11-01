"""Environment verification for the HOI-GNN project.
Run from the project root:  python -m utils.env_check
"""
import sys
import time


def main() -> int:
    print(f"Python        : {sys.version.split()[0]}")
    ok = True

    import torch
    import torchvision
    print(f"torch         : {torch.__version__}")
    print(f"torchvision   : {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        print(f"GPU           : {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        print(f"CUDA runtime  : {torch.version.cuda}")
    else:
        print("WARNING: running on CPU fallback.")

    import torch_geometric
    from torch_geometric.nn import GATv2Conv, GCNConv, GATConv
    from torch_geometric.data import Data, Batch
    from torch_geometric.explain import Explainer, GNNExplainer
    print(f"torch_geometric: {torch_geometric.__version__}")

    d1 = Data(x=torch.randn(3, 8), edge_index=torch.tensor([[0, 1], [1, 2]]),
              edge_attr=torch.randn(2, 4))
    d2 = Data(x=torch.randn(2, 8), edge_index=torch.tensor([[0], [1]]),
              edge_attr=torch.randn(1, 4))
    batch = Batch.from_data_list([d1, d2]).to(device)
    conv = GATv2Conv(8, 16, heads=2, edge_dim=4).to(device)
    out = conv(batch.x, batch.edge_index, batch.edge_attr)
    assert out.shape == (5, 32), f"unexpected shape {out.shape}"
    print(f"GATv2Conv batch forward OK, output shape {tuple(out.shape)}")

    import ultralytics
    print(f"ultralytics   : {ultralytics.__version__}")

    import captum, cv2, numpy, pandas, sklearn, networkx, matplotlib, streamlit, pycocotools
    print(f"captum        : {captum.__version__}")
    print(f"opencv        : {cv2.__version__}")
    print(f"numpy         : {numpy.__version__}")
    print(f"pandas        : {pandas.__version__}")
    print(f"scikit-learn  : {sklearn.__version__}")
    print(f"networkx      : {networkx.__version__}")
    print(f"matplotlib    : {matplotlib.__version__}")
    print(f"streamlit     : {streamlit.__version__}")
    print("pycocotools   : imported OK")

    if numpy.__version__.startswith("2"):
        print("WARNING: numpy 2.x detected; run: pip install \"numpy==1.26.4\"")
        ok = False

    if device == "cuda":
        t0 = time.time()
        a = torch.randn(2048, 2048, device="cuda")
        _ = a @ a
        torch.cuda.synchronize()
        print(f"CUDA matmul test OK ({time.time() - t0:.2f}s)")

    print("\nALL CHECKS PASSED" if ok else "\nCHECKS COMPLETED WITH WARNINGS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())