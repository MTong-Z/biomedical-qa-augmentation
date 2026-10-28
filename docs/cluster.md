# GPU Execution

GPU resources have not yet been inspected and no GPU job has been submitted. GPU training is an optional reader extension, outside the core method and prototype release. The included chat demonstration can be validated without either an API key or a GPU. Readers may optionally use an API teacher to generate additional candidates without a local teacher deployment.

After logging into the university cluster and completing its authentication process, run `bash scripts/inspect_cluster.sh` to inspect permitted accounts and partitions. A login node may have no visible GPU; run inference and training only within a GPU allocation.

Create a project environment under a writable server project or scratch directory and install `.[gpu]`. Fix the teacher and student model identities, commit hashes and generation settings before running. Use `scripts/run_gpu_job.sh` as a command wrapper when submitting to the scheduler. Account, partition, GPU count, memory and time limits must be set from the owner's actual allocation rather than assumed defaults.

Start with one source example for each route and inspect output, context length, GPU memory and speed. Then run the 100-source pilot. Cache files and checkpoints belong in the run directory and should survive session disconnects. Capture the dependency freeze and the scheduler job record alongside the experiment manifest.
