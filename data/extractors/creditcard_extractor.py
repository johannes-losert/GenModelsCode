from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel, Field
import os
import logging
import inspect

class CreditCardDescription(BaseModel):
    """Model for a credit card description with validation."""
    description: str = Field(
        min_length=1,
        description="Full text description of the credit card including all features and benefits"
    )

class CreditCardExtractor(ABC):
    """Abstract base class defining the interface for credit card information extraction."""

    @property
    @abstractmethod
    def site_name(self) -> str:
        """Name of the website being extracted from."""
        pass
    
    def extract(self) -> List[CreditCardDescription]:
        """
        Extract credit card descriptions from HTML file in implementing class's directory.
        Assumes there is only one HTML file in the directory.
        
        Returns:
            List[CreditCardDescription]: List of validated credit card descriptions
        """
        impl_file = inspect.getfile(self.__class__)
        impl_dir = os.path.dirname(os.path.abspath(impl_file))
        html_files = [f for f in os.listdir(impl_dir) if f.endswith('.html')]
        
        if not html_files:
            raise FileNotFoundError(f"No HTML file found in directory: {impl_dir}")
        if len(html_files) > 1:
            raise ValueError(f"Multiple HTML files found in directory: {impl_dir}. Expected only one HTML file.")
        
        html_file = os.path.join(impl_dir, html_files[0])
        
        # Read the HTML content
        with open(html_file, 'r', encoding='utf-8') as file:
            html_content = file.read()
            
        return self._process_html(html_content)
    
    @abstractmethod
    def _process_html(self, html_content: str) -> List[CreditCardDescription]:
        """
        Process the HTML content to extract credit card descriptions.
        To be implemented by concrete classes.
        
        Args:
            html_content (str): Raw HTML content containing credit card information
            
        Returns:
            List[CreditCardDescription]: List of validated credit card descriptions
        """
        pass