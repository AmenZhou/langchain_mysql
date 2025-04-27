from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SchemaExtractor:
    def __init__(self, engine):
        self.engine = engine

    def get_all_tables(self) -> List[str]:
        """Get all table names from the database."""
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except SQLAlchemyError as e:
            logger.error(f"Error getting table names: {e}")
            return []

    async def extract_table_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Extract schema information for a specific table or all tables."""
        try:
            inspector = inspect(self.engine)
            
            if table_name:
                # Get columns for specific table
                columns = inspector.get_columns(table_name)
                column_info = []
                for column in columns:
                    column_info.append({
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column.get("nullable", True),
                        "default": str(column.get("default", "None")),
                        "primary_key": column.get("primary_key", False)
                    })
                
                # Get primary key
                pk = inspector.get_pk_constraint(table_name)
                
                # Get foreign keys
                fks = inspector.get_foreign_keys(table_name)
                
                # Get indexes
                indexes = inspector.get_indexes(table_name)
                
                # Combine all schema information
                return {
                    "table_name": table_name,
                    "columns": column_info,
                    "primary_key": pk,
                    "foreign_keys": fks,
                    "indexes": indexes
                }
            else:
                # Get schema for all tables
                schema_info = {}
                for table in self.get_all_tables():
                    columns = inspector.get_columns(table)
                    schema_info[table] = {
                        'columns': [col['name'] for col in columns],
                        'description': f"Table containing {', '.join(col['name'] for col in columns)}"
                    }
                return schema_info
            
        except SQLAlchemyError as e:
            logger.error(f"Error extracting schema for table {table_name}: {e}")
            return {"table_name": table_name, "error": str(e)} if table_name else {}

    def create_schema_documents(self, schema_info: Dict) -> List[Document]:
        """Create Document objects for each table schema to be embedded."""
        if not schema_info:
            logger.warning("No schema information provided.")
            return []
        
        documents = []
        
        for table_name, table_info in schema_info.items():
            # Create an extremely minimal description
            description = f"{table_name}:"
            
            # Add only column names, skip types and other details
            columns = table_info.get('columns', [])
            
            # Join columns with spaces to save even more space
            description += " ".join(columns)
            
            # Create metadata with minimal schema information
            metadata = {
                "table_name": table_name,
                "columns": columns
            }
            
            # Create the document
            doc = Document(page_content=description, metadata=metadata)
            documents.append(doc)
        
        return documents

    def create_prompt_documents(self) -> List[Document]:
        """Create Document objects for prompts to be embedded."""
        from .prompts import PROMPT_REFINE, PROMPT_TABLE_QUERY, get_sanitize_prompt
        
        prompt_docs = []
        
        # Add refine prompt
        prompt_docs.append(Document(
            page_content=PROMPT_REFINE,
            metadata={"prompt_type": "refine"}
        ))
        
        # Add table query prompt
        prompt_docs.append(Document(
            page_content=PROMPT_TABLE_QUERY,
            metadata={"prompt_type": "table"}
        ))
        
        # Add sanitize prompt
        prompt_docs.append(Document(
            page_content=get_sanitize_prompt(""),
            metadata={"prompt_type": "sanitize"}
        ))
        
        return prompt_docs 
