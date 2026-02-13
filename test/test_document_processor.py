# test_document_processor.py
import os
import sys
from pathlib import Path

# Añadir la carpeta 'src' al PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from sonika_ai_toolkit.document_processing.processor import DocumentProcessor

def create_test_files():
    """Crea archivos de prueba si no existen"""
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    # Crear archivo TXT de prueba
    txt_file = test_dir / "test.txt"
    if not txt_file.exists():
        txt_file.write_text(
            "Este es un documento de prueba.\n"
            "Contiene múltiples líneas de texto.\n"
            "Será usado para probar el DocumentProcessor.\n" * 10,
            encoding='utf-8'
        )
    
    return test_dir


def test_token_counting():
    """Prueba el conteo de tokens"""
    print("\n" + "="*60)
    print("TEST 1: Conteo de tokens")
    print("="*60)
    
    test_text = "Este es un texto de prueba para contar tokens."
    token_count = DocumentProcessor.count_tokens(test_text)
    
    print(f"Texto: {test_text}")
    print(f"Tokens contados: {token_count}")
    
    assert token_count > 0, "El conteo de tokens debe ser mayor a 0"
    print("✅ Test de conteo de tokens: PASSED")


def test_txt_extraction():
    """Prueba extracción de texto TXT"""
    print("\n" + "="*60)
    print("TEST 2: Extracción de texto TXT")
    print("="*60)
    
    test_dir = create_test_files()
    txt_file = test_dir / "test.txt"
    
    try:
        text = DocumentProcessor.extract_text(str(txt_file), "txt")
        print(f"Texto extraído ({len(text)} caracteres):")
        print(text[:200] + "..." if len(text) > 200 else text)
        
        assert len(text) > 0, "El texto extraído no debe estar vacío"
        assert "documento de prueba" in text.lower(), "El texto debe contener el contenido esperado"
        print("✅ Test de extracción TXT: PASSED")
        
        return text
    except Exception as e:
        print(f"❌ Test de extracción TXT: FAILED - {str(e)}")
        raise


def test_chunking(text):
    """Prueba la creación de chunks"""
    print("\n" + "="*60)
    print("TEST 3: Creación de chunks")
    print("="*60)
    
    try:
        chunks = DocumentProcessor.create_chunks(
            text=text,
            chunk_size=100,  # Más pequeño para testing
            overlap=20
        )
        
        print(f"Número de chunks generados: {len(chunks)}")
        
        assert len(chunks) > 0, "Debe generar al menos un chunk"
        
        # Verificar estructura de cada chunk
        for i, chunk in enumerate(chunks[:3]):  # Mostrar solo primeros 3
            print(f"\n--- Chunk {i} ---")
            print(f"Index: {chunk['chunk_index']}")
            print(f"Tokens: {chunk['token_count']}")
            print(f"Content: {chunk['content'][:100]}...")
            
            assert 'content' in chunk, "Chunk debe tener 'content'"
            assert 'chunk_index' in chunk, "Chunk debe tener 'chunk_index'"
            assert 'token_count' in chunk, "Chunk debe tener 'token_count'"
            assert 'metadata' in chunk, "Chunk debe tener 'metadata'"
            assert chunk['chunk_index'] == i, "Los índices deben ser secuenciales"
        
        print(f"\n✅ Test de chunking: PASSED ({len(chunks)} chunks generados)")
        
        return chunks
    except Exception as e:
        print(f"❌ Test de chunking: FAILED - {str(e)}")
        raise


def test_unsupported_format():
    """Prueba manejo de formato no soportado"""
    print("\n" + "="*60)
    print("TEST 4: Formato no soportado")
    print("="*60)
    
    try:
        DocumentProcessor.extract_text("test.xyz", "xyz")
        print("❌ Test de formato no soportado: FAILED - Debería haber lanzado ValueError")
        assert False, "Debería haber lanzado ValueError"
    except ValueError as e:
        print(f"Error esperado capturado: {str(e)}")
        assert "not supported" in str(e).lower(), "El mensaje de error debe indicar formato no soportado"
        print("✅ Test de formato no soportado: PASSED")


def test_pdf_extraction_optional():
    """Prueba extracción de PDF si existe"""
    print("\n" + "="*60)
    print("TEST 5: Extracción de PDF (opcional)")
    print("="*60)
    
    test_pdf = "test_documents/sample.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"⚠️  No se encontró {test_pdf}")
        print("Para probar PDF, coloca un archivo PDF en test_documents/sample.pdf")
        print("✅ Test de PDF: SKIPPED")
        return
    
    try:
        text = DocumentProcessor.extract_text(test_pdf, "pdf")
        print(f"Texto extraído de PDF ({len(text)} caracteres):")
        print(text[:200] + "..." if len(text) > 200 else text)
        
        assert len(text) > 0, "El texto extraído del PDF no debe estar vacío"
        print("✅ Test de extracción PDF: PASSED")
    except ImportError as e:
        print(f"⚠️  PyPDF2 no instalado: {str(e)}")
        print("Instala con: pip install PyPDF2")
        print("✅ Test de PDF: SKIPPED")
    except Exception as e:
        print(f"❌ Test de extracción PDF: FAILED - {str(e)}")


def test_docx_extraction_optional():
    """Prueba extracción de DOCX si existe"""
    print("\n" + "="*60)
    print("TEST 6: Extracción de DOCX (opcional)")
    print("="*60)
    
    test_docx = "test_documents/sample.docx"
    
    if not os.path.exists(test_docx):
        print(f"⚠️  No se encontró {test_docx}")
        print("Para probar DOCX, coloca un archivo DOCX en test_documents/sample.docx")
        print("✅ Test de DOCX: SKIPPED")
        return
    
    try:
        text = DocumentProcessor.extract_text(test_docx, "docx")
        print(f"Texto extraído de DOCX ({len(text)} caracteres):")
        print(text[:200] + "..." if len(text) > 200 else text)
        
        assert len(text) > 0, "El texto extraído del DOCX no debe estar vacío"
        print("✅ Test de extracción DOCX: PASSED")
    except ImportError as e:
        print(f"⚠️  python-docx no instalado: {str(e)}")
        print("Instala con: pip install python-docx")
        print("✅ Test de DOCX: SKIPPED")
    except Exception as e:
        print(f"❌ Test de extracción DOCX: FAILED - {str(e)}")


def run_all_tests():
    """Ejecuta todos los tests"""
    print("\n" + "🧪" * 30)
    print("INICIANDO TESTS DE DocumentProcessor")
    print("🧪" * 30)
    
    try:
        # Tests obligatorios
        test_token_counting()
        text = test_txt_extraction()
        test_chunking(text)
        test_unsupported_format()
        
        # Tests opcionales (si hay archivos)
        test_pdf_extraction_optional()
        test_docx_extraction_optional()
        
        # Resumen
        print("\n" + "="*60)
        print("RESUMEN DE TESTS")
        print("="*60)
        print("✅ Todos los tests obligatorios: PASSED")
        print("\nPara probar más formatos:")
        print("1. Coloca un PDF en: test_documents/sample.pdf")
        print("2. Coloca un DOCX en: test_documents/sample.docx")
        print("3. Ejecuta de nuevo este script")
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TESTS FALLIDOS")
        print("="*60)
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)