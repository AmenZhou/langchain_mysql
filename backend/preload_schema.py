#!/usr/bin/env python3
import logging
import argparse
from .schema_vectorizer import preload_schema_to_vectordb

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Preload MySQL schema into vector database')
    parser.add_argument('--persist-dir', type=str, default='./chroma_db',
                        help='Directory to persist the vector database')
    args = parser.parse_args()
    
    logger.info(f"Starting schema preloading to {args.persist_dir}")
    try:
        preload_schema_to_vectordb(persist_directory=args.persist_dir)
        logger.info("Schema successfully preloaded to vector database!")
    except Exception as e:
        logger.error(f"Error preloading schema: {e}")
        raise

if __name__ == "__main__":
    main() 
