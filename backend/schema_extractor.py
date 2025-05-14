from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SchemaExtractor:
    def __init__(self, engine, inspector=None):
        self.engine = engine
        self._inspector = inspector

    @property
    def inspector(self):
        if self._inspector is None:
            if self.engine is None:
                raise ValueError("Cannot create inspector without an engine if one is not provided.")
            self._inspector = inspect(self.engine)
        return self._inspector

    def get_all_tables(self) -> List[str]:
        """Get all table names from the database."""
        try:
            return self.inspector.get_table_names()
        except SQLAlchemyError as e:
            logger.error(f"Error getting table names: {e}")
            return []

    async def extract_table_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Extract schema information for a specific table or all tables."""
        try:
            logger.info("Starting schema extraction")
            current_inspector = self.inspector
            
            if table_name:
                logger.info(f"Extracting schema for specific table: {table_name}")
                # Get columns for specific table
                columns = current_inspector.get_columns(table_name)
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
                pk = current_inspector.get_pk_constraint(table_name)
                
                # Get foreign keys
                fks = current_inspector.get_foreign_keys(table_name)
                
                # Get indexes
                indexes = current_inspector.get_indexes(table_name)
                
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
                # First pass: collect all table information
                for table in tables:
                    logger.info(f"Processing table: {table}")
                    try:
                        columns = current_inspector.get_columns(table)
                        column_info = []
                        for column in columns:
                            column_info.append({
                                "name": column["name"],
                                "type": str(column["type"]),
                                "description": f"Column {column['name']} of type {str(column['type'])}",
                                "nullable": column.get("nullable", True),
                                "primary_key": column.get("primary_key", False)
                            })
                        
                        # Get foreign keys
                        fks = current_inspector.get_foreign_keys(table)
                        foreign_key_info = []
                        for fk in fks:
                            foreign_key_info.append({
                                "column": fk["constrained_columns"][0],
                                "references": f"{fk['referred_table']}.{fk['referred_columns'][0]}",
                                "description": f"Links to {fk['referred_table']} through {fk['referred_columns'][0]}"
                            })
                        
                        schema_info[table] = {
                            'columns': column_info,
                            'foreign_keys': foreign_key_info,
                            'description': f"Table {table} contains {', '.join(col['name'] for col in columns)}"
                        }
                    except Exception as table_error:
                        logger.error(f"Error processing table {table}: {str(table_error)}")
                        continue

                # Second pass: enhance descriptions with relationship information
                for table, info in schema_info.items():
                    # Add relationship descriptions
                    relationship_desc = []
                    
                    # Add outgoing relationships (foreign keys)
                    if info['foreign_keys']:
                        fk_desc = []
                        for fk in info['foreign_keys']:
                            ref_table = fk['references'].split('.')[0]
                            fk_desc.append(f"links to {ref_table} through {fk['column']}")
                        relationship_desc.append(f"This table {', '.join(fk_desc)}")
                    
                    # Add incoming relationships (tables that reference this table)
                    incoming_refs = []
                    for other_table, other_info in schema_info.items():
                        if other_table != table:
                            for fk in other_info['foreign_keys']:
                                if fk['references'].split('.')[0] == table:
                                    incoming_refs.append(f"{other_table} references this table through {fk['column']}")
                    if incoming_refs:
                        relationship_desc.append(f"This table is referenced by: {', '.join(incoming_refs)}")
                    
                    # Update the description with relationship information
                    if relationship_desc:
                        info['description'] += "\n" + "\n".join(relationship_desc)
                    
                    # Add a business logic description
                    if table == 'film':
                        info['description'] += "\nThis table contains movie information including title, description, and rental details."
                    elif table == 'actor':
                        info['description'] += "\nThis table contains actor information including their names."
                    elif table == 'film_actor':
                        info['description'] += "\nThis table links actors to films, creating a many-to-many relationship."
                    elif table == 'inventory':
                        info['description'] += "\nThis table tracks physical copies of films available for rent."
                    elif table == 'rental':
                        info['description'] += "\nThis table records when films are rented and returned."
                    elif table == 'payment':
                        info['description'] += "\nThis table records payments made for film rentals."
                
                logger.info(f"Successfully extracted schema for {len(schema_info)} tables")
                return schema_info
            
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error extracting schema: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error extracting schema: {e}")
            return {}

    def create_schema_documents(self, schema_info: Dict) -> List[Document]:
        """Create Document objects for each table schema to be embedded."""
        if not schema_info:
            logger.warning("No schema information provided.")
            return []
        
        documents = []
        
        for table_name, table_info in schema_info.items():
            # Create a more descriptive document content
            content_parts = []
            
            # Add table name and basic description
            content_parts.append(f"Table {table_name} contains {', '.join(col.get('name', '') for col in table_info.get('columns', []))}")
            
            # Add column information with types and constraints
            content_parts.append("\nColumns:")
            columns = table_info.get('columns', [])
            for col in columns:
                col_name = col.get('name', '')
                col_type = col.get('type', '')
                col_desc = col.get('description', '')
                constraints = []
                if col.get('primary_key', False):
                    constraints.append("PRIMARY KEY")
                if not col.get('nullable', True):
                    constraints.append("NOT NULL")
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                content_parts.append(f"- {col_name} ({col_type}){constraint_str}: {col_desc}")
            
            # Add foreign key relationships
            foreign_keys = table_info.get('foreign_keys', [])
            if foreign_keys:
                content_parts.append("\nForeign Key Relationships:")
                for fk in foreign_keys:
                    content_parts.append(f"- {fk['column']} references {fk['references']}: {fk['description']}")
            
            # Create the document with enhanced content
            doc = Document(
                page_content="\n".join(content_parts),
                metadata={
                    "table_name": table_name,
                    "columns": [col.get('name', '') for col in columns],
                    "foreign_keys": foreign_keys,
                    "description": table_info.get('description', '')
                }
            )
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
