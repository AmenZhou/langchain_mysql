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
            logger.info("Starting schema extraction")
            inspector = inspect(self.engine)
            
            if table_name:
                logger.info(f"Extracting schema for specific table: {table_name}")
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
                logger.info("Extracting schema for all tables")
                tables = self.get_all_tables()
                logger.info(f"Found {len(tables)} tables: {tables}")
                
                schema_info = {}
                for table in tables:
                    logger.info(f"Processing table: {table}")
                    try:
                        columns = inspector.get_columns(table)
                        column_info = []
                        for column in columns:
                            column_info.append({
                                "name": column["name"],
                                "type": str(column["type"]),
                                "description": f"Column {column['name']} of type {str(column['type'])}"
                            })
                        
                        # Get foreign keys
                        fks = inspector.get_foreign_keys(table)
                        foreign_key_info = []
                        for fk in fks:
                            foreign_key_info.append({
                                "column": fk["constrained_columns"][0],
                                "references": f"{fk['referred_table']}.{fk['referred_columns'][0]}"
                            })
                        
                        # Create table description with relationships
                        description = f"Table {table} contains {', '.join(col['name'] for col in columns)}"
                        if foreign_key_info:
                            fk_desc = " and is linked to: " + ", ".join(
                                f"{fk['column']} references {fk['references']}" 
                                for fk in foreign_key_info
                            )
                            description += fk_desc
                        
                        schema_info[table] = {
                            'columns': column_info,
                            'foreign_keys': foreign_key_info,
                            'description': description
                        }
                        logger.info(f"Successfully processed table {table} with {len(columns)} columns and {len(foreign_key_info)} foreign keys")
                    except Exception as table_error:
                        logger.error(f"Error processing table {table}: {str(table_error)}")
                        continue
                
                logger.info(f"Successfully extracted schema for {len(schema_info)} tables")
                return schema_info
            
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error extracting schema for table {table_name}: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            return {"table_name": table_name, "error": str(e)} if table_name else {}
        except Exception as e:
            logger.error(f"Unexpected error extracting schema for table {table_name}: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No additional details'}")
            return {"table_name": table_name, "error": str(e)} if table_name else {}

    def create_schema_documents(self, schema_info: Dict) -> List[Document]:
        """Create Document objects for each table schema to be embedded."""
        if not schema_info:
            logger.warning("No schema information provided.")
            return []
        
        documents = []
        
        for table_name, table_info in schema_info.items():
            # Create a more descriptive document content
            description = f"Table {table_name} contains "
            
            # Add column information with types
            columns = table_info.get('columns', [])
            column_descriptions = []
            for col in columns:
                col_name = col.get('name', '')
                col_type = col.get('type', '')
                col_desc = col.get('description', '')
                column_descriptions.append(f"{col_name} ({col_type}) - {col_desc}")
            
            # Add column descriptions to the document content
            description += " and ".join(column_descriptions)
            
            # Create metadata with full schema information
            metadata = {
                "table_name": table_name,
                "columns": columns,
                "description": description
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
