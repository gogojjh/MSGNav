# MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation

<div align="center">
  <img src="teaser.png", width="800" />
</div>

## 📰 News
- [Coming Soon] 🤖 Deployment code for Unitree G1 will be released.
- [Mar 4, 2026] 🚀 Code is released.
- [Feb 21, 2026] 🎉 MSGNav is accepted to CVPR 2026.

## Installation

```bash
conda create -n msgnav python=3.9 -y && conda activate msgnav
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# or: pip install torch==2.0.1 torchvision==0.15.2 --trusted-host http://pypi.tuna.tsinghua.edu.cn
conda install -c conda-forge -c aihabitat habitat-sim=0.2.5 headless faiss-cpu=1.7.4 -y
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2 -y
pip install omegaconf==2.3.0 open-clip-torch==2.26.1 ultralytics==8.2.31 supervision==0.21.0 opencv-python-headless==4.10.0.84 \
 scikit-learn==1.4 scikit-image==0.22 open3d==0.18.0 hipart==1.0.4 openai==1.35.3 httpx==0.27.2 numpy==1.24.3 scipy==1.11.4 ollama
```
