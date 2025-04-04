import cv2
import dlib
import numpy as np
import sqlite3

DB_PATH = "control_acceso.db"
FACE_DETECTOR = dlib.get_frontal_face_detector()
FACE_ENCODER = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
SHAPE_PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# 📌 Inicializar base de datos SQLite
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            tarjeta_rfid TEXT UNIQUE,
            vector BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()






# 📌 Capturar rostro y obtener vector facial
def capturar_vector():
    cap = cv2.VideoCapture(0)
    vectores = []

    print("[INFO] Capturando imágenes... Presiona 'ESPACIO' para tomar una foto.")

    while len(vectores) < 5:  # Capturar mínimo 5 imágenes
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_DETECTOR(gray)

        for face in faces:
            landmarks = SHAPE_PREDICTOR(gray, face)
            vector = FACE_ENCODER.compute_face_descriptor(frame, landmarks)
            vectores.append(np.array(vector))  # Convertir a numpy array

        cv2.imshow("Captura", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32 and len(faces) > 0:  # Tecla ESPACIO y hay una cara detectada
            break

    cap.release()
    cv2.destroyAllWindows()

    return np.mean(vectores, axis=0)  # Vector promedio

# 📌 Registrar usuario con su vector facial
def registrar_usuario(nombre, tarjeta_rfid):
    vector_facial = capturar_vector()
    vector_bytes = vector_facial.tobytes()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO usuarios (nombre, tarjeta_rfid, vector) VALUES (?, ?, ?)", 
                   (nombre, tarjeta_rfid, vector_bytes))
    conn.commit()
    conn.close()

    print(f"[INFO] Usuario {nombre} registrado con éxito.")

# 📌 Verificar acceso comparando vectores
def verificar_acceso():
    vector_facial = capturar_vector()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre, vector FROM usuarios")
    usuarios = cursor.fetchall()
    
    for usuario in usuarios:
        usuario_id, nombre, vector_bytes = usuario
        vector_db = np.frombuffer(vector_bytes, dtype=np.float64)  # Convertir de bytes a numpy array
        diferencia = np.linalg.norm(vector_facial - vector_db)  # Distancia euclidiana

        if diferencia < 0.6:  # Umbral ajustable (mejor precisión con Dlib)
            print(f"✅ Acceso permitido: {nombre}")
            conn.close()
            return
    
    print("🚫 Acceso denegado")
    conn.close()

def mostrar_usuarios():
    conn = sqlite3.connect("control_acceso.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id, nombre, tarjeta_rfid FROM usuarios")
    usuarios = cursor.fetchall()

    for usuario in usuarios:
        print(f"ID: {usuario[0]}, Nombre: {usuario[1]}, RFID: {usuario[2]}")

    conn.close()

# 📌 Menú principal
if __name__ == "__main__":
    init_db()
    
    while True:
        print("\n--- SISTEMA DE CONTROL DE ACCESO ---")
        print("1. Registrar nuevo usuario")
        print("2. Verificar acceso")
        print("3.Ver usuarios registrados")
        print("4. Salir")
        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            nombre = input("Nombre del usuario: ")
            tarjeta_rfid = input("Número de tarjeta RFID: ")
            registrar_usuario(nombre, tarjeta_rfid)
        elif opcion == "2":
            verificar_acceso()
        elif opcion == "3":
            mostrar_usuarios()
        elif opcion == "4":
            break
        else:
            print("❌ Opción inválida. Intenta de nuevo.")
