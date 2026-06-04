# Runbook

## First Login

```bash
ssh -p 22 root@36.139.133.196
```

After the public key is installed, prefer key auth:

```bash
ssh -i ~/.ssh/id_ed25519 -p 22 root@36.139.133.196
```

Rotate or disable the temporary password after the key path is confirmed.

## Host Checks

```bash
nvidia-smi
nvidia-smi topo -m
df -h
free -h
python3 --version
docker --version
tmux -V
```

## Tmux

```bash
tmux new -s rl
tmux attach -t rl
```

Use `/data/sdb/rl-posttrain/logs` for training logs.

## If 32B OOMs

Reduce in this order:

1. `max_response_length`
2. `num_generations`
3. `max_prompt_length`
4. LoRA rank
5. vLLM memory utilization

If it still fails, switch the main result to `Qwen/Qwen3-14B`.

