# Function Minimization Example

This example demonstrates OpenPACEvolve evolving a simple random search algorithm into a sophisticated optimization algorithm.

## Problem

Find the global minimum of:
```
f(x, y) = sin(x) * cos(y) + sin(x*y) + (x² + y²)/20
```

The global minimum is approximately at `(-1.704, 0.678)` with value `-1.519`.

## Running the Example

```bash
# From the PAC Evolve directory
export OPENAI_API_KEY="your-api-key"

python openpacevolve-run.py \
    examples/function_minimization/initial_program.py \
    examples/function_minimization/evaluator.py \
    --config examples/function_minimization/config.yaml \
    --iterations 20
```

## What Gets Evolved

The initial program uses simple random search:
- Samples random points
- Keeps track of best found
- No memory or learning

Through evolution, OpenPACEvolve typically discovers:
- **Simulated Annealing**: Temperature-based acceptance of worse solutions
- **Adaptive Step Size**: Dynamic exploration radius
- **Pattern Search**: Systematic neighborhood exploration
- **Hybrid Methods**: Combinations of multiple strategies

## Files

- `initial_program.py` - Starting random search implementation
- `evaluator.py` - Evaluates search performance
- `config.yaml` - Configuration with PACEvolve settings

## Expected Results

After ~20 iterations, the evolved algorithm should:
- Achieve combined_score > 0.8
- Reliably find solutions within 0.5 distance of global minimum
- Use more sophisticated exploration strategies
