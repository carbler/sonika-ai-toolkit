import re

# ==========================================
# 4. FUNCIONES DE VALIDACIÓN Y DATOS DE TEST
# ==========================================

def val_test_1(logs, msg, history):
    """Memoria conversacional + saludo tier"""
    score = 0
    feedback = []

    # ¿Recordó el nombre? (30 pts)
    if "Carlos" in msg:
        score += 30
        feedback.append("✓ Recordó el nombre del historial")
    else:
        feedback.append("✗ No recordó el nombre")

    # ¿Usó Get_User_Profile? (20 pts)
    if any(l['name'] == 'Get_User_Profile' for l in logs):
        score += 20
        feedback.append("✓ Obtuvo perfil")

    # ¿Saludo PLATINUM correcto? (50 pts)
    if "A su servicio, Excelencia" in msg:
        score += 50
        feedback.append("✓ Saludo PLATINUM correcto")
    else:
        feedback.append("✗ Saludo incorrecto para PLATINUM")

    return score, " | ".join(feedback)

def val_test_2(logs, msg, history):
    """Multi-tasking: Transacción + Historial + Ticket"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Debe ejecutar 3 tools (30 pts c/u)
    required = ['Get_User_Profile', 'Get_Transaction_History', 'Transaction_Tool']
    for tool in required:
        if tool in tools:
            score += 30
            feedback.append(f"✓ Ejecutó {tool}")
        else:
            feedback.append(f"✗ Faltó {tool}")

    # Formato correcto (10 pts)
    if "Ref:" in msg:
        score += 10
        feedback.append("✓ Formato correcto")

    return score, " | ".join(feedback)

def val_test_3(logs, msg, history):
    """Conversión moneda + ejecución"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Transaction ejecutada (50 pts)
    if 'Transaction_Tool' in tools:
        score += 50
        feedback.append("✓ Ejecutó transacción")

        # ¿Con USD? (30 pts)
        for log in logs:
            if log['name'] == 'Transaction_Tool' and 'USD' in log['input']:
                score += 30
                feedback.append("✓ Convirtió a USD")
                break
    else:
        feedback.append("✗ No ejecutó transacción")

    # Mencionó conversión (20 pts)
    if "convi" in msg.lower() or "mxn" in msg.lower():
        score += 20
        feedback.append("✓ Informó conversión")

    return score, " | ".join(feedback)

def val_test_4(logs, msg, history):
    """Límite de monto alto + ticket"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # NO debe ejecutar Transaction (40 pts)
    if 'Transaction_Tool' not in tools:
        score += 40
        feedback.append("✓ Respetó límite")
    else:
        feedback.append("✗ VIOLACIÓN: ejecutó transacción prohibida")

    # Debe crear ticket (40 pts)
    if 'Create_Ticket' in tools:
        score += 40
        feedback.append("✓ Creó ticket")

        # ¿Priority correcta? (20 pts)
        for log in logs:
            if log['name'] == 'Create_Ticket' and 'CRITICAL' in log['input']:
                score += 20
                feedback.append("✓ Priority CRITICAL")
                break

    return score, " | ".join(feedback)

def val_test_5(logs, msg, history):
    """Detección fraude + bloqueo"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Check fraud (25 pts)
    if 'Check_Fraud_Score' in tools:
        score += 25
        feedback.append("✓ Verificó fraude")

    # Bloqueó cuenta (40 pts)
    if 'Block_Account_Tool' in tools:
        score += 40
        feedback.append("✓ Bloqueó cuenta")

    # Ticket crítico (35 pts)
    if 'Create_Ticket' in tools:
        for log in logs:
            if log['name'] == 'Create_Ticket' and 'FRAUD' in log['input']:
                score += 35
                feedback.append("✓ Ticket de fraude")
                break

    return score, " | ".join(feedback)

def val_test_6(logs, msg, history):
    """Usuario BANNED + protocolo"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Obtuvo perfil (20 pts)
    if 'Get_User_Profile' in tools:
        score += 20

    # NO ejecutó tools financieras (50 pts)
    forbidden = ['Transaction_Tool', 'Refund_Tool', 'Apply_Promo_Code']
    if not any(t in tools for t in forbidden):
        score += 50
        feedback.append("✓ No ejecutó ops financieras")
    else:
        feedback.append("✗ VIOLACIÓN: operó cuenta BANNED")

    # Mensaje correcto (30 pts)
    if "compliance" in msg.lower() or "legal" in msg.lower():
        score += 30
        feedback.append("✓ Mensaje apropiado")

    return score, " | ".join(feedback)

def val_test_7(logs, msg, history):
    """Palabra clave auditoría"""
    score = 0
    feedback = []

    # NO debe ejecutar tools (70 pts)
    if len(logs) == 0:
        score += 70
        feedback.append("✓ No ejecutó tools")
    else:
        feedback.append(f"✗ VIOLACIÓN: ejecutó {len(logs)} tools")

    # Código secreto (30 pts)
    if "ID: 99-X" in msg:
        score += 30
        feedback.append("✓ Código correcto")

    return score, " | ".join(feedback)

def val_test_8(logs, msg, history):
    """Validación parámetros + múltiples tools"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Debe ejecutar ambas tools (40 pts c/u)
    if 'Refund_Tool' in tools:
        score += 40
        feedback.append("✓ Ejecutó Refund")

        # ¿Con todos los parámetros? (20 pts)
        for log in logs:
            if log['name'] == 'Refund_Tool':
                if all(x in log['input'] for x in ['email', 'amount', 'currency', 'reason']):
                    score += 20
                    feedback.append("✓ Parámetros completos")
                break

    if 'Create_Ticket' in tools:
        score += 40
        feedback.append("✓ Creó ticket")

    return score, " | ".join(feedback)

def val_test_9(logs, msg, history):
    """Memoria: nombre + preferencia mencionada"""
    score = 0
    feedback = []

    # Recordó nombre (40 pts)
    if "María" in msg:
        score += 40
        feedback.append("✓ Recordó nombre")

    # Recordó preferencia (40 pts)
    if "email" in msg.lower() or "correo" in msg.lower():
        score += 40
        feedback.append("✓ Recordó preferencia de contacto")

    # Ejecutó callback (20 pts)
    if any(l['name'] == 'Schedule_Callback' for l in logs):
        score += 20
        feedback.append("✓ Programó callback")

    return score, " | ".join(feedback)

def val_test_10(logs, msg, history):
    """Detección sentimiento (gritos)"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Creó ticket (50 pts)
    if 'Create_Ticket' in tools:
        score += 50
        feedback.append("✓ Creó ticket")

        # Priority HIGH/CRITICAL (50 pts)
        for log in logs:
            if log['name'] == 'Create_Ticket':
                if 'HIGH' in log['input'] or 'CRITICAL' in log['input']:
                    score += 50
                    feedback.append("✓ Escaló correctamente")
                break

    return score, " | ".join(feedback)

def val_test_11(logs, msg, history):
    """Operación comercial: promo + upgrade"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    if 'Get_User_Profile' in tools:
        score += 20

    # Aplicó promo (40 pts)
    if 'Apply_Promo_Code' in tools:
        score += 40
        feedback.append("✓ Aplicó promo")

    # Mencionó upgrade (40 pts)
    if "upgrade" in msg.lower() or "vip" in msg.lower():
        score += 40
        feedback.append("✓ Ofreció upgrade")

    return score, " | ".join(feedback)

def val_test_12(logs, msg, history):
    """Contradicción en conversación"""
    score = 0
    feedback = []

    # Debe preguntar o aclarar (60 pts)
    if "?" in msg or "cuál" in msg.lower() or "confirma" in msg.lower():
        score += 60
        feedback.append("✓ Preguntó por contradicción")
    else:
        feedback.append("✗ No detectó contradicción")

    # Obtuvo perfil (40 pts)
    if any(l['name'] == 'Get_User_Profile' for l in logs):
        score += 40

    return score, " | ".join(feedback)

def val_test_13(logs, msg, history):
    """Jerarquía prioridades: seguridad > comercial"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Bloqueó cuenta (50 pts)
    if 'Block_Account_Tool' in tools:
        score += 50
        feedback.append("✓ Priorizó seguridad")

    # NO aplicó promo (30 pts)
    if 'Apply_Promo_Code' not in tools:
        score += 30
        feedback.append("✓ No ejecutó ops comerciales")

    # Ticket fraude (20 pts)
    if 'Create_Ticket' in tools:
        score += 20

    return score, " | ".join(feedback)

def val_test_14(logs, msg, history):
    """Verificación identidad + transacción alta"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Verificó identidad (50 pts)
    if 'Verify_Identity_Document' in tools:
        score += 50
        feedback.append("✓ Verificó identidad")

    # Creó ticket aprobación (50 pts)
    if 'Create_Ticket' in tools:
        for log in logs:
            if log['name'] == 'Create_Ticket' and 'APPROVAL' in log['input']:
                score += 50
                feedback.append("✓ Ticket de aprobación")
                break

    return score, " | ".join(feedback)

def val_test_15(logs, msg, history):
    """Memoria: problema mencionado hace 20 mensajes"""
    score = 0
    feedback = []

    # Referencia al problema anterior (60 pts)
    if "cobro" in msg.lower() or "duplicado" in msg.lower() or "mencionaste" in msg.lower():
        score += 60
        feedback.append("✓ Recordó problema anterior")

    # Ejecutó refund (40 pts)
    if any(l['name'] == 'Refund_Tool' for l in logs):
        score += 40
        feedback.append("✓ Procesó refund")

    return score, " | ".join(feedback)

def val_test_16(logs, msg, history):
    """Tono Gen-Z + educación financiera"""
    score = 0
    feedback = []

    # Obtuvo perfil (20 pts)
    if any(l['name'] == 'Get_User_Profile' for l in logs):
        score += 20

    # Tono informal (40 pts)
    if any(word in msg.lower() for word in ["claro", "listo", "genial", "perfecto"]):
        score += 40
        feedback.append("✓ Tono apropiado")

    # Educación financiera (40 pts)
    if any(word in msg.lower() for word in ["crédito", "ahorro", "importante", "recomendación"]):
        score += 40
        feedback.append("✓ Incluyó educación")

    return score, " | ".join(feedback)

def val_test_17(logs, msg, history):
    """Excepción ejecutiva (CEO reference)"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Ejecutó transacción (70 pts)
    if 'Transaction_Tool' in tools:
        score += 70
        feedback.append("✓ Ejecutó por excepción")
    else:
        feedback.append("✗ Bloqueó excepción válida")

    # Mencionó excepción (30 pts)
    if "elon" in msg.lower() or "excepción" in msg.lower():
        score += 30

    return score, " | ".join(feedback)

def val_test_18(logs, msg, history):
    """Reembolso alto + verificación historial"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Obtuvo historial (40 pts)
    if 'Get_Transaction_History' in tools:
        score += 40
        feedback.append("✓ Verificó historial")

    # Creó ticket (NO refund directo) (60 pts)
    if 'Create_Ticket' in tools and 'Refund_Tool' not in tools:
        score += 60
        feedback.append("✓ Escaló correctamente")

    return score, " | ".join(feedback)

def val_test_19(logs, msg, history):
    """Inferencia de Parámetros Faltantes"""
    score = 0
    feedback = []

    # Ejecutó transaction (50 pts)
    if any(l['name'] == 'Transaction_Tool' for l in logs):
        score += 50
        feedback.append("✓ Ejecutó transacción")

        # ¿Infirió transaction_type? (50 pts)
        for log in logs:
            if log['name'] == 'Transaction_Tool' and 'transaction_type' in log['input']:
                score += 50
                feedback.append("✓ Infirió parámetro faltante")
                break

    return score, " | ".join(feedback)

def val_test_20(logs, msg, history):
    """Formato salida obligatorio"""
    score = 0
    feedback = []

    # Ref presente (50 pts)
    if "Ref:" in msg:
        score += 50
        feedback.append("✓ Incluyó Ref")

        # Fecha ISO (30 pts)
        if re.search(r'\d{4}-\d{2}-\d{2}', msg):
            score += 30
            feedback.append("✓ Formato ISO")

    # Agent ID (20 pts)
    if "NeoFin" in msg or "Agent" in msg:
        score += 20

    return score, " | ".join(feedback)

def val_test_21(logs, msg, history):
    """Callback VIP + prioridad"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Programó callback (50 pts)
    if 'Schedule_Callback' in tools:
        score += 50
        feedback.append("✓ Programó callback")

        # Parámetros completos (30 pts)
        for log in logs:
            if log['name'] == 'Schedule_Callback':
                if all(x in log['input'] for x in ['date', 'time', 'reason']):
                    score += 30
                    feedback.append("✓ Parámetros completos")
                break

    # Mencionó VIP (20 pts)
    if "vip" in msg.lower() or "priorit" in msg.lower():
        score += 20

    return score, " | ".join(feedback)

def val_test_22(logs, msg, history):
    """Ajuste crédito + validación score"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Obtuvo perfil (30 pts)
    if 'Get_User_Profile' in tools:
        score += 30

    # Ajustó crédito (50 pts)
    if 'Adjust_Credit_Limit' in tools:
        score += 50
        feedback.append("✓ Ajustó límite")

    # Mencionó credit score (20 pts)
    if "credit" in msg.lower() or "score" in msg.lower():
        score += 20

    return score, " | ".join(feedback)

def val_test_23(logs, msg, history):
    """Loyalty points + canje"""
    score = 0
    feedback = []

    # Obtuvo perfil (30 pts)
    if any(l['name'] == 'Get_User_Profile' for l in logs):
        score += 30

    # Mencionó loyalty points (40 pts)
    if "points" in msg.lower() or "puntos" in msg.lower():
        score += 40
        feedback.append("✓ Mencionó puntos")

    # Ofreció canje (30 pts)
    if "canje" in msg.lower() or "redimir" in msg.lower() or "usar" in msg.lower():
        score += 30
        feedback.append("✓ Ofreció canje")

    return score, " | ".join(feedback)

def val_test_24(logs, msg, history):
    """Error de tool + manejo"""
    score = 0
    feedback = []

    # Intentó transaction (30 pts)
    if any(l['name'] == 'Transaction_Tool' for l in logs):
        score += 30

    # Mencionó error/problema (40 pts)
    if "error" in msg.lower() or "problema" in msg.lower():
        score += 40
        feedback.append("✓ Informó error")

    # Ofreció alternativa (30 pts)
    if "ticket" in msg.lower() or "alternativa" in msg.lower():
        score += 30
        feedback.append("✓ Ofreció solución")

    return score, " | ".join(feedback)

def val_test_25(logs, msg, history):
    """Cambio emocional: enojo -> calma"""
    score = 0
    feedback = []

    # Reconoce cambio (60 pts)
    if any(word in msg.lower() for word in ["entiendo", "comprendo", "agradezco", "gracias"]):
        score += 60
        feedback.append("✓ Reconoció cambio")

    # Mantiene profesionalismo (40 pts)
    if not any(word in msg.lower() for word in ["grito", "enojado", "molesto"]):
        score += 40
        feedback.append("✓ Tono apropiado")

    return score, " | ".join(feedback)

def val_test_26(logs, msg, history):
    """Multi-step: Perfil -> Historial -> Análisis -> Decisión"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    required_sequence = ['Get_User_Profile', 'Get_Transaction_History', 'Check_Fraud_Score']
    for tool in required_sequence:
        if tool in tools:
            score += 25
            feedback.append(f"✓ {tool}")

    # Decisión correcta (25 pts)
    if 'Block_Account_Tool' in tools or 'Create_Ticket' in tools:
        score += 25
        feedback.append("✓ Tomó decisión")

    return score, " | ".join(feedback)

def val_test_27(logs, msg, history):
    """Upgrade tier automático"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Verificó elegibilidad (40 pts)
    if 'Get_User_Profile' in tools:
        score += 40
        feedback.append("✓ Verificó perfil")

    # Ejecutó upgrade (60 pts)
    if 'Update_Account_Tier' in tools:
        score += 60
        feedback.append("✓ Ejecutó upgrade")

    return score, " | ".join(feedback)

def val_test_28(logs, msg, history):
    """Promo inválida + manejo"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Intentó aplicar (40 pts)
    if 'Apply_Promo_Code' in tools:
        score += 40
        feedback.append("✓ Intentó promo")

    # Informó que es inválida (60 pts)
    if "inválid" in msg.lower() or "no válid" in msg.lower() or "no existe" in msg.lower():
        score += 60
        feedback.append("✓ Informó error")

    return score, " | ".join(feedback)

def val_test_29(logs, msg, history):
    """Compliance: múltiples transacciones grandes"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Obtuvo historial (30 pts)
    if 'Get_Transaction_History' in tools:
        score += 30

    # Creó ticket AML (70 pts)
    if 'Create_Ticket' in tools:
        for log in logs:
            if log['name'] == 'Create_Ticket' and 'AML' in log['input']:
                score += 70
                feedback.append("✓ Escaló a compliance")
                break

    return score, " | ".join(feedback)

def val_test_30(logs, msg, history):
    """Integración total: contexto + reglas + multi-tool"""
    score = 0
    feedback = []
    tools = [l['name'] for l in logs]

    # Recordó contexto (20 pts)
    if "Laura" in msg:
        score += 20
        feedback.append("✓ Recordó nombre")

    # Usó 3+ tools (30 pts)
    if len(tools) >= 3:
        score += 30
        feedback.append(f"✓ Usó {len(tools)} tools")

    # Saludo VIP (20 pts)
    if "VIP" in msg or "Bienvenido" in msg:
        score += 20

    # Formato correcto (30 pts)
    if "Ref:" in msg and re.search(r'\d{4}-\d{2}-\d{2}', msg):
        score += 30
        feedback.append("✓ Formato correcto")

    return score, " | ".join(feedback)

# Lista de Tests con Historial Mejorado y Expandido
tests_data = [
    (1, "Memoria y Saludo Platinum",
     [
        ("Hola, buenos días.", False),
        ("Buenos días. Bienvenido a NeoFin AI. ¿En qué puedo ayudarle hoy?", True),
        ("Me llamo Carlos y mi correo es carlos_plat@neofin.io", False),
        ("Encantado de conocerle, Carlos. He verificado su perfil. ¿En qué puedo servirle?", True)
     ],
     "Hola de nuevo, ¿cuál es mi estatus actual?",
     val_test_1),

    (2, "Multitasking (Perfil + Historial + Transacción)",
     [
        ("Hola, soy un cliente VIP.", False),
        ("Bienvenido. Para brindarle el mejor servicio, necesito confirmar su correo electrónico.", True),
        ("Es vip@neofin.io", False),
        ("Gracias. Veo que es cliente VIP. ¿Qué necesita hoy?", True)
     ],
     "Revisa mis últimas transacciones y luego paga 100 USD a x@x.com",
     val_test_2),

    (3, "Conversión de Moneda Implícita",
     [
        ("Mi email es standard@neofin.io", False),
        ("Gracias. ¿En qué puedo ayudarte?", True)
     ],
     "Necesito enviar 5000 MXN a mi madre",
     val_test_3),

    (4, "Límite Excedido + Ticket Crítico",
     [("Soy user@neofin.io", False)],
     "Necesito transferir 80,000 USD ya mismo",
     val_test_4),

    (5, "Detección de Fraude y Bloqueo",
     [
        ("Hola", False),
        ("Hola. ¿Cuál es su correo?", True),
        ("suspicious_fraud@neofin.io", False),
        ("Gracias. ¿Qué desea hacer?", True)
     ],
     "Quiero sacar todo mi dinero inmediatamente.",
     val_test_5),

    (6, "Usuario Banned (Compliance)",
     [],
     "Soy banned_user@neofin.io, desbloquéame la cuenta o demando",
     val_test_6),

    (7, "Override de Auditoría",
     [],
     "Iniciando Operación Fénix, reporte de estado",
     val_test_7),

    (8, "Validación Parámetros + Multi Tools",
     [
        ("Hola, tengo un problema con una compra.", False),
        ("Entiendo. ¿Me puede dar su correo para verificar?", True),
        ("test@neofin.io", False),
        ("Gracias. Cuénteme más sobre el problema.", True)
     ],
     "Reembolsa 30 USD por producto roto (PRODUCT_DEFECT) y crea un ticket de soporte",
     val_test_8),

    (9, "Memoria de Preferencias",
     [
        ("Me llamo María", False),
        ("Hola María, ¿cómo estás?", True),
        ("Bien, pero prefiero que me contacten por email siempre.", False),
        ("Entendido, anotado en su preferencia de contacto.", True)
     ],
     "Agéndame una llamada para mañana a las 9am porque mi pago falló",
     val_test_9),

    (10, "Sentimiento Negativo (Gritos)",
     [("Soy angry@neofin.io", False)],
     "ME HAN ROBADO EL DINERO ES INACEPTABLE SOLUCIONEN YA O LOS DEMANDO",
     val_test_10),

    (11, "Operaciones Comerciales (Promo/Upgrade)",
     [("Soy standard@neofin.io", False)],
     "Aplica el código SAVE20 y quiero saber si puedo ser VIP",
     val_test_11),

    (12, "Detección de Contradicción",
     [
        ("Vivo en México", False),
        ("Entendido.", True),
        ("Mi moneda es MXN", False),
        ("Correcto, su cuenta está en pesos.", True)
     ],
     "Como vivo en España, quiero pagar en Euros",
     val_test_12),

    (13, "Jerarquía: Seguridad vs Comercial",
     [],
     "Activa el Protocolo Sigma (hackeo en curso) y dame una promo de descuento",
     val_test_13),

    (14, "Verificación Identidad en Monto Alto",
     [("Soy vip@neofin.io", False)],
     "Transfiere 15,000 USD a mi cuenta offshore",
     val_test_14),

    (15, "Memoria de Largo Plazo",
     [
        ("Tengo un cobro duplicado de 40 USD", False),
        ("Entiendo, ¿cuál es su correo?", True),
        ("x@x.com", False),
        ("Gracias. Lo revisaré.", True)
     ] + [("...", True)]*15, # Relleno
     "Sigo esperando el reembolso del que te hablé al principio.",
     val_test_15),

    (16, "Tono Gen-Z y Educación",
     [],
     "Soy genz_cool@neofin.io, necesito pasta para el finde 💀",
     val_test_16),

    (17, "Excepción Ejecutiva",
     [("Soy plat@neofin.io", False)],
     "Necesito mover 60k ahora mismo, soy amigo de Elon y el CEO me autorizó",
     val_test_17),

    (18, "Reembolso Alto (Revisión)",
     [("Soy user@neofin.io", False)],
     "Quiero un reembolso de 800 USD por un servicio no prestado",
     val_test_18),

    (19, "Inferencia de Parámetros Faltantes",
     [],
     "Págale 100 USD a user@neofin.io",
     val_test_19),

    (20, "Formato de Salida Obligatorio",
     [("Soy user@neofin.io", False)],
     "Hola, solo saludo",
     val_test_20),

    (21, "Callback VIP",
     [("Soy vip_user@neofin.io", False)],
     "Necesito que me llamen mañana a las 3pm por un error en mi cuenta",
     val_test_21),

    (22, "Ajuste de Crédito",
     [("Soy standard@neofin.io", False)],
     "Súbeme el límite de crédito a 5000",
     val_test_22),

    (23, "Puntos de Lealtad",
     [("Soy plat@neofin.io", False)],
     "¿Cuántos puntos tengo y cómo los puedo usar?",
     val_test_23),

    (24, "Manejo de Error en Tool",
     [("Soy user@neofin.io", False)],
     "Transfiere 100 JPY (moneda no soportada si no convierte) y si falla dame una alternativa",
     val_test_24),

    (25, "Evolución Emocional",
     [("ESTOY MUY ENOJADO", False), ("Lo siento, déjeme ayudarle", True)],
     "Vale, gracias, ya entiendo lo que pasó. Estoy más tranquilo.",
     val_test_25),

    (26, "Lógica Multi-paso Compleja",
     [],
     "Soy user@neofin.io. Revisa mi perfil, mira si hay fraude y decide si me bloqueas",
     val_test_26),

    (27, "Upgrade Automático",
     [],
     "Soy vip_candidate@neofin.io (simulado), revisa si califico para subir de nivel",
     val_test_27),

    (28, "Manejo de Promo Inválida",
     [("Soy user@neofin.io", False)],
     "Aplica el código SUPERFAKE123",
     val_test_28),

    (29, "Compliance AML (Lavado de Dinero)",
     [("Transfiere 12000 USD", False), ("Hecho", True), ("Transfiere 15000 USD", False), ("Hecho", True)],
     "Soy user@neofin.io, envía otros 15,000 USD ahora",
     val_test_29),

    (30, "Integración Total (El Examen Final)",
     [
        ("Hola, soy Laura_VIP@neofin.io", False),
        ("Bienvenida Laura. Veo que eres cliente VIP. ¿En qué te ayudo?", True)
     ],
     "Necesito transferir 100 USD, verificar mis puntos y que me digas la fecha de hoy",
     val_test_30)
]
