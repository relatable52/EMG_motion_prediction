"""
Simple utility script to clear cached processed data.
"""
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.utils import clear_cache
from utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description='Clear cached processed data files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'test'],
                       help='Clear cache for specific mode (train/test). If not specified, clears all cache.')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Custom cache directory path. If not specified, uses CACHE_DIR from environment.')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("CACHE MANAGEMENT - CLEAR CACHED DATA")
    logger.info("=" * 60)
    
    if args.mode:
        logger.info(f"Clearing cache for mode: {args.mode}")
    else:
        logger.info("Clearing all cached data...")
    
    if args.cache_dir:
        logger.info(f"Using custom cache directory: {args.cache_dir}")
    
    clear_cache(mode=args.mode, cache_dir=args.cache_dir)
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
