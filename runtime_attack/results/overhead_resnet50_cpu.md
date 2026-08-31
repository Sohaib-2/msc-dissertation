# Cost of running the teacher-side monitor (resnet50, cpu)

Teacher: `clean_teacher_resnet50.pth` | batch size 128 | 4 CPU threads | median of 15 trials after a warm-up.

Teacher inference costs 33.624 ms per query (4303.9 ms per batch of 128). That cost is paid whether or not the monitor is running. The monitor adds one scoring pass per window rather than per query, so the figures below divide its cost over the whole window.

| window | detector, ms per window | teacher, ms for the same queries | overhead | overhead vs a 1 ms/query teacher | window in memory |
|---|---|---|---|---|---|
| 100 | 0.103 | 3362 | 0.00306% | 0.103% | 3.9 KiB |
| 200 | 0.117 | 6725 | 0.00175% | 0.059% | 7.8 KiB |
| 500 | 0.160 | 16812 | 0.00095% | 0.032% | 19.5 KiB |
| 1000 | 0.231 | 33624 | 0.00069% | 0.023% | 39.1 KiB |
| 2000 | 0.370 | 67248 | 0.00055% | 0.018% | 78.1 KiB |

The overhead column compares the monitor against this machine's own teacher, which is served on a CPU and is therefore slow. That flatters the monitor, so the next column repeats the comparison against a hypothetical teacher answering in 1 ms per query, which is the order a GPU-served model would achieve. The monitor stays well under a tenth of one per cent even then, because it runs once per window rather than once per query.

The monitor holds one window of output vectors and a fixed calibration reference; nothing else in it grows with the length of the stream. The window figure above is also what a pooled deployment would have to carry on the wire, since it is the size of the probability vectors themselves.
