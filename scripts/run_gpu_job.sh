#!/usr/bin/env bash
# Submit this wrapper with the account, partition, memory and time permitted by your cluster.
# Example arguments: python -m biomedqa.cli augment ...
set -euo pipefail
if [[ $# -eq 0 ]]; then
    printf 'Pass the experiment command as arguments.\n' >&2
    exit 2
fi
python -c 'import torch; assert torch.cuda.is_available(), "No allocated CUDA GPU"; print(torch.cuda.get_device_name(0))'
exec "$@"
