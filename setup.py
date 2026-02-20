from setuptools import setup, find_packages

setup(
    name="sonika-ai-toolkit",
    version="0.2.7",
    description="Toolkit para creación de agentes de IA y procesamiento de documentos",
    author="Erley Blanco Carvajal",
    license="MIT License",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),  # Encuentra los paquetes dentro de "src"
    package_dir={"": "src"},  # Indica que los paquetes están en el directorio "src"
    include_package_data=True,  # Importante para incluir archivos no-Python definidos en MANIFEST.in
    install_requires=[
        "langchain-mcp-adapters==0.1.9",
        "langchain-community==0.3.26",
        "langchain-core==0.3.66",
        "langchain-openai==0.3.24",
        "langgraph==0.4.8",
        "langgraph-checkpoint==2.1.0",
        "langgraph-sdk==0.1.70",
        "dataclasses-json==0.6.7",
        "python-dateutil==2.9.0.post0",
        "pydantic==2.11.7",
        "faiss-cpu==1.11.0",
        "pypdf==5.6.1",
        "python-dotenv==1.0.1",
        "typing_extensions==4.14.0",
        "typing-inspect==0.9.0",
        "PyPDF2==3.0.1",
        "python-docx==1.2.0",
        "openpyxl==3.1.5",
        "python-pptx==1.0.2",
        "nest-asyncio==1.6.0"
    ],

    extras_require={
        "dev": [
            "sphinx>=8.1.3,<9.0.0",
            "sphinx-rtd-theme>=3.0.1,<4.0.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Verifica la versión mínima de Python compatible
)
