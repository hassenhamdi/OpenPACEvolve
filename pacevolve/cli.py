"""
Command-line interface for OpenPACEvolve.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from pacevolve._version import __version__
from pacevolve.config import load_config
from pacevolve.controller import Controller


def setup_logging(level: str, log_dir: str = None) -> None:
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_dir:
        log_path = Path(log_dir) / "openpacevolve.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers,
    )


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="OpenPACEvolve - Progress-Aware Consistent Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "initial_program",
        help="Path to initial program file",
    )
    parser.add_argument(
        "evaluator",
        help="Path to evaluator module",
    )
    parser.add_argument(
        "--config", "-c",
        dest="config_path",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        help="Number of evolution iterations",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_dir",
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint",
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"OpenPACEvolve {__version__}",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.initial_program).exists():
        print(f"Error: Initial program not found: {args.initial_program}")
        return 1
    
    if not Path(args.evaluator).exists():
        print(f"Error: Evaluator not found: {args.evaluator}")
        return 1
    
    # Load config
    config = load_config(args.config_path)
    
    # Override with CLI arguments
    if args.iterations:
        config.max_iterations = args.iterations
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.log_level:
        config.log_level = args.log_level
    
    # Setup logging
    setup_logging(config.log_level, config.log_dir)
    
    logger = logging.getLogger(__name__)
    logger.info(f"OpenPACEvolve {__version__}")
    logger.info(f"Initial program: {args.initial_program}")
    logger.info(f"Evaluator: {args.evaluator}")
    logger.info(f"Iterations: {config.max_iterations}")
    
    # Create controller
    controller = Controller(
        initial_program_path=args.initial_program,
        evaluator_path=args.evaluator,
        config=config,
        output_dir=args.output_dir,
    )
    
    # Run evolution
    try:
        best = asyncio.run(controller.run())
        
        if best:
            print(f"\n{'='*60}")
            print(f"Evolution Complete!")
            print(f"Best Score: {best.score:.4f}")
            print(f"Best Program: {config.output_dir}/best_program.py")
            print(f"{'='*60}")
            return 0
        else:
            print("Evolution completed but no valid programs found.")
            return 1
            
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user.")
        return 130
    except Exception as e:
        logger.exception(f"Evolution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
