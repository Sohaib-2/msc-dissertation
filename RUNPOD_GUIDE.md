# Running the experiments on a rented GPU

The results in the dissertation were produced on rented GPUs. This is the procedure, kept here
so the runs can be repeated.

| Script | What it produces | Approximate cost | Approximate time |
|---|---|---|---|
| `run_runpod.sh` | The attack and detector results at full scale | $1-3 | 2 h |
| `run_teacher_repro.sh` | An independently trained teacher, to check the detector results hold on a second draw | under $1 | 25 min |
| `run_scar_full.sh` | The SCAR baseline at the paper's own settings | $8-10 | 15-25 h |

The first two were run. The third was not: the cost-bounded reproduction in `scar_repro/` already
established the baseline mechanism, and the full run was left as optional work.

## Procedure

1. Rent an instance with an RTX 3090 or better and a PyTorch 2.x image. 50 GB of disk is enough.

2. Open a terminal and start `tmux`, so that the run survives a dropped connection:

   ```bash
   tmux new -s run
   ```

3. Fetch the code:

   ```bash
   git clone https://github.com/Sohaib-2/msc-dissertation.git
   cd msc-dissertation
   git clone https://github.com/WhitolfChen/SCAR.git scar_baseline_upstream
   cp -r scar_baseline_upstream/* scar_baseline/
   ```

   The SCAR harness is not redistributed here; see the note on attribution in the top-level
   README.

4. Run one of the scripts:

   ```bash
   bash run_runpod.sh
   bash run_teacher_repro.sh
   ```

   Each writes its logs, its result files and a `SUMMARY.txt` into a dated directory under
   `runtime_attack/results/`. Detach from `tmux` with `Ctrl-b d` and reattach with
   `tmux attach -t run`.

5. Copy the results directory back before terminating the instance:

   ```bash
   rsync -avz -e "ssh -p <port> -i ~/.ssh/id_ed25519" \
     root@<host>:/workspace/msc-dissertation/runtime_attack/results/ ./results/
   ```

   Rename any directory whose filenames would collide with existing results before copying it
   back. The reproduction run writes the same filenames as the original run, and copying it over
   the top would destroy the results the dissertation cites.

6. Terminate the instance. Nothing on it is preserved, so confirm that every file arrived at the
   expected size first.

## Notes

- Use the instance's direct TCP address rather than a proxy host for `rsync`.
- On some images `pip` needs `--break-system-packages`.
- `run_teacher_repro.sh` prints the figures the original run produced in its own header, so the
  two can be compared without going back to the dissertation.
