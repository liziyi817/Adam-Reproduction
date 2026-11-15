# Adam reproduction on UCI Optdigits

Reimplements Algorithm 1 of Kingma and Ba (ICLR 2015) and compares it
with momentum SGD and Adam without bias correction on the same MLP.
The original UCI files are downloaded on demand.

Tested with Python 3.9.15 and PyTorch 2.7.1 CPU on Windows.
Run from this directory:

    python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt
    python run.py --verify

Verification compares four float64 Adam updates with PyTorch Adam and
checks the saved test predictions for all nine weight files.
Each checkpoint contains the MLP state dict. The seed and optimizer
are encoded in its file name. CSV rows follow the UCI optdigits.tes order.

    python run.py --output rerun
    python plot.py --results rerun

Retraining repeats validation searches and paired final runs in a new folder.
The delivered metrics.json records data file hashes, tuning scores, train
curves, final test scores and confusion matrices.
