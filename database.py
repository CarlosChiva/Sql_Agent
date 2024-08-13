import sqlite3

# Conectar a la base de datos SQLite
con = sqlite3.connect('mi_base_de_datos.db')
cur = con.cursor()

# Crear tabla empleados
cur.execute("""
    CREATE TABLE IF NOT EXISTS empleados (
        id INTEGER PRIMARY KEY,
        nombre TEXT NOT NULL,
        apellidos TEXT NOT NULL
    );
""")

# Crear tabla clientes
cur.execute("""
    CREATE TABLE IF NOT EXISTS clientes (
        id INTEGER PRIMARY KEY,
        nombre TEXT NOT NULL,
        empleado_id INTEGER NOT NULL,
        FOREIGN KEY (empleado_id) REFERENCES empleados(id)
    );
""")
try:
    # Completar con algunos datos de ejemplo
    cur.execute("INSERT INTO empleados (id, nombre, apellidos) VALUES (1, 'Juan', 'Pérez')")
    cur.execute("INSERT INTO empleados (id, nombre, apellidos) VALUES (2, 'Maria', 'González')")

    cur.execute("INSERT INTO clientes (id, nombre, empleado_id) VALUES (1, 'Cliente 1', 1)")
    cur.execute("INSERT INTO clientes (id, nombre, empleado_id) VALUES (2, 'Cliente 2', 1)")  # Asignado al mismo empleado
    cur.execute("INSERT INTO clientes (id, nombre, empleado_id) VALUES (3, 'Cliente 3', 2)")
    # Guardar los cambios en la base de datos
    con.commit()
except sqlite3.IntegrityError as e:
    print(f"Error de integridad: {e}")


# Cerrar la conexión a la base de datos
con.close()

def get_tables():
    con = sqlite3.connect('mi_base_de_datos.db')
    # Crea un cursor para ejecutar las consultas
    cur = con.cursor()

    # Consulta que devuelve el nombre de todas las tablas
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")

    # Obtiene los resultados
    tablas = cur.fetchall()

    # Cierra la conexión a la base de datos
    con.close()

    table_list=[]
    # Imprime el nombre de cada tabla
    for tabla in tablas:
        table_list.append(tabla[0])
    return table_list


