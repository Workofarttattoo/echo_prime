#---
description: Deploy Echo/Kraton, run demo, push to Hugging Face, and execute benchmarks
#---

1. Start the dashboard server
```bash
nohup python /Users/noone/echo_prime/dashboard_server.py > dashboard.log 2>&1 &
```

2. Run the hive‑mind demo
```bash
python /Users/noone/echo_prime/demo_hive_mind.py
```

3. Push the latest code to the Hugging Face Space (Update URL if needed)
```bash
# git remote set-url hf https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE
git push hf HEAD:main
```

4. Execute the benchmark suite
```bash
# Run lightweight sample first
python3 /Users/noone/echo_prime/full_benchmark_runner.py --datasets gsm8k --samples 5

# Or run full suite (requires powerful hardware)
# bash /Users/noone/echo_prime/execute_full_benchmarks.sh
```
