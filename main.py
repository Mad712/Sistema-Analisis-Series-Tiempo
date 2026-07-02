"""
Sistema: SAIE - Sistema de Análisis de Indicadores Económicos

Usuarios de prueba:
  correo: analista@saie.com  | contraseña: 1234
  correo: admin@saie.com     | contraseña: admin2024
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =============================================================
# ENTIDADES
# =============================================================

class EUsuario:
    """Diagrama 1: E-Usuario"""

    def __init__(self, correo: str, contrasena: str, rol: str):
        self.usuario   = correo
        self.contrasena = contrasena
        self.rol       = rol

    def validar_datos(self, correo: str, contrasena: str) -> bool:
        return self.usuario == correo and self.contrasena == contrasena

    def enviar(self, exitoso: bool) -> str:
        if exitoso:
            return f"Acceso concedido. Bienvenido, {self.rol}."
        return "Credenciales incorrectas. Intente nuevamente."


class EDatosHistoricos:
    """Diagrama 2: E-Datos históricos"""

    def __init__(self):
        self.serie    = None
        self.columna  = None
        self.ruta     = None

    def guardar_datos(self, serie, columna: str, ruta: str):
        self.serie   = serie
        self.columna = columna
        self.ruta    = ruta

    def mostrar_mensaje(self) -> str:
        if self.serie is not None:
            return (f"Datos cargados: {len(self.serie)} observaciones | "
                    f"Columna: {self.columna} | "
                    f"Período: {self.serie.index[0].year}–{self.serie.index[-1].year}")
        return "Sin datos cargados."


class ESES:
    """Diagrama 3: E-SES"""

    def __init__(self):
        self.resultados = None

    def guardar_resultados(self, resultados):
        self.resultados = resultados


class EArima:
    """Diagrama 4: E-Arima"""

    def __init__(self):
        self.ajuste     = None
        self.pronostico = None

    def guardar_resultados(self, ajuste, pronostico):
        self.ajuste     = ajuste
        self.pronostico = pronostico


# =============================================================
# BASE DE DATOS SIMULADA
# =============================================================

class _BaseDeDatos:
    def __init__(self):
        self.__registros = [
            EUsuario("analista@saie.com", "1234",      "Analista de Datos"),
            EUsuario("admin@saie.com",    "admin2024", "Administrador"),
        ]

    def buscar(self, correo: str):
        for u in self.__registros:
            if u.usuario == correo:
                return u
        return None


# =============================================================
# CONTROLADORES
# =============================================================

class CCLogin:
    """Diagrama 1: CC-Login"""

    def __init__(self):
        self.__bd = _BaseDeDatos()

    def validar_datos(self, correo: str, contrasena: str) -> dict:
        vacios = []
        if not correo.strip():
            vacios.append("Usuario")
        if not contrasena.strip():
            vacios.append("Contraseña")

        if vacios:
            return {
                "exitoso": False,
                "mensaje": f"Campos vacíos: {' y '.join(vacios)}.",
                "usuario": None,
            }

        entidad = self.__bd.buscar(correo)
        if entidad is None or not entidad.validar_datos(correo, contrasena):
            return {
                "exitoso": False,
                "mensaje": "Credenciales incorrectas. Intente nuevamente.",
                "usuario": None,
            }

        return {
            "exitoso": True,
            "mensaje": entidad.enviar(True),
            "usuario": entidad,
        }

    def enviar(self, mensaje: str) -> None:
        print(f"\n  {mensaje}\n")


class CCIngresarDatosHistoricos:
    """Diagrama 2: CC-Ingresar datos históricos"""

    def __init__(self):
        self.__entidad = EDatosHistoricos()

    def __leer_archivo_raw(self, ruta: str):
        """Lee el archivo crudo según su extensión."""
        ext = os.path.splitext(ruta)[1].lower()

        if ext in [".xlsx", ".xls"]:
            for i in range(20):
                df_temp = pd.read_excel(ruta, header=i, nrows=1)
                cols = [str(c).strip().lower() for c in df_temp.columns]
                if any(c in ["año", "ano", "year"] for c in cols):
                    return pd.read_excel(ruta, header=i).dropna(how="all")
            return pd.read_excel(ruta, header=3).dropna(how="all")

        elif ext == ".csv":
            for sep in [",", ";", "	", "|"]:
                for enc in ["utf-8", "latin-1"]:
                    try:
                        df = pd.read_csv(ruta, sep=sep, encoding=enc)
                        if len(df.columns) > 1:
                            return df.dropna(how="all")
                    except Exception:
                        continue
            raise ValueError("No se pudo leer el CSV.")

        elif ext == ".xml":
            return pd.read_xml(ruta).dropna(how="all")

        else:
            raise ValueError(f"Formato no soportado: {ext}. Use .xlsx, .csv o .xml")

    def __es_formato_banco_mundial(self, df) -> bool:
        """Detecta si el archivo tiene formato Banco Mundial (países en filas, años en columnas)."""
        cols = [str(c).strip() for c in df.columns]
        # Banco Mundial tiene columnas: Country Name, Country Code, Indicator Name, Indicator Code, 1960, 1961...
        tiene_country = any(c.lower() in ["country name", "country code"] for c in cols)
        tiene_anios   = any(str(c).strip().isdigit() and 1900 <= int(c) <= 2100
                            for c in cols if str(c).strip().isdigit())
        return tiene_country and tiene_anios

    def __transponer_banco_mundial(self, df) -> dict:
        """
        Convierte el formato Banco Mundial (países en filas, años en columnas)
        a formato vertical (Año, Valor) para el país seleccionado.
        Retorna lista de países disponibles y el df original.
        """
        cols = [str(c).strip() for c in df.columns]

        # Encontrar columna de nombre de país
        col_pais = None
        for c in df.columns:
            if str(c).strip().lower() == "country name":
                col_pais = c
                break

        # Encontrar columnas de años
        cols_anios = [c for c in df.columns
                      if str(c).strip().isdigit()
                      and 1900 <= int(str(c).strip()) <= 2100]

        paises = df[col_pais].dropna().unique().tolist()
        return {
            "col_pais":   col_pais,
            "cols_anios": cols_anios,
            "paises":     paises,
            "df_raw":     df,
        }

    def validar_datos(self, ruta: str) -> dict:
        if not os.path.exists(ruta):
            return {"exitoso": False, "mensaje": "Archivo no encontrado.", "entidad": None}

        try:
            df = self.__leer_archivo_raw(ruta)

            # ── Formato Banco Mundial (multi-país) ──
            if self.__es_formato_banco_mundial(df):
                info = self.__transponer_banco_mundial(df)
                return {
                    "exitoso":           True,
                    "mensaje":           "Archivo Banco Mundial detectado.",
                    "entidad":           None,
                    "formato":           "banco_mundial",
                    "df":                df,
                    "col_pais":          info["col_pais"],
                    "cols_anios":        info["cols_anios"],
                    "paises":            info["paises"],
                }

            # ── Formato estándar (Año en columna) ──
            col_anio = None
            for col in df.columns:
                if str(col).strip().lower() in ["año", "ano", "year"]:
                    col_anio = str(col).strip()
                    break
            if col_anio is None:
                for col in df.columns:
                    muestra = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(muestra) > 0 and muestra.between(1900, 2100).all():
                        col_anio = str(col).strip()
                        break

            if col_anio is None:
                return {"exitoso": False,
                        "mensaje": "No se detectó columna de años.",
                        "entidad": None}

            columnas_num = [
                c for c in df.columns
                if str(c).strip() != col_anio
                and str(c) not in ["nan", ""]
                and not str(c).startswith("Unnamed")
                and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
            ]

            if not columnas_num:
                return {"exitoso": False,
                        "mensaje": "No se encontraron columnas numéricas.",
                        "entidad": None}

            return {
                "exitoso":      True,
                "mensaje":      "Archivo válido.",
                "entidad":      None,
                "formato":      "estandar",
                "df":           df,
                "col_anio":     col_anio,
                "columnas_num": columnas_num,
            }

        except Exception as e:
            return {"exitoso": False, "mensaje": f"Error al leer archivo: {e}", "entidad": None}

    def cargar_desde_banco_mundial(self, df, col_pais, cols_anios, pais, ruta) -> EDatosHistoricos:
        """Extrae la fila del país seleccionado y la convierte en serie temporal."""
        fila = df[df[col_pais] == pais].iloc[0]

        datos = {}
        for col in cols_anios:
            val = fila[col]
            try:
                val_num = float(val)
                if not pd.isna(val_num):
                    datos[int(str(col).strip())] = val_num
            except (ValueError, TypeError):
                continue

        serie = pd.Series(datos)
        serie.index = pd.to_datetime(serie.index, format="%Y")
        serie.name = pais

        self.__entidad.guardar_datos(serie, pais, ruta)
        return self.__entidad

    def mostrar_mensaje(self, resultado: dict) -> str:
        return resultado["mensaje"]

    def cargar_serie(self, df, col_anio: str, columna: str, ruta: str):
        # Limpiar espacios en nombres de columnas del df
        df.columns = [str(c).strip() for c in df.columns]
        col_anio   = col_anio.strip()
        columna    = columna.strip()

        df_s = df[[col_anio, columna]].copy()
        df_s = df_s.dropna(subset=[col_anio, columna])
        df_s = df_s[df_s[col_anio].astype(str).str.match(r"^\d{4}\.?0*$")]
        df_s["Fecha"] = pd.to_datetime(df_s[col_anio].astype(int), format="%Y")
        df_s = df_s.set_index("Fecha")
        serie = df_s[columna].astype(float)

        self.__entidad.guardar_datos(serie, columna, ruta)
        return self.__entidad

    def get_entidad(self):
        return self.__entidad


class CCAplicarSES:
    """Diagrama 3: CC-Aplicar SES"""

    def __init__(self):
        self.__entidad = ESES()

    def validar_valor_alfa(self, alpha) -> bool:
        return 0 < alpha < 1

    def calcular_ses(self, datos, alpha, periodos):
        if not self.validar_valor_alfa(alpha):
            self.enviar_mensaje("Error: α debe estar entre 0 y 1.")
            return None, None

        modelo     = SimpleExpSmoothing(datos).fit(smoothing_level=alpha, optimized=False)
        suavizado  = modelo.fittedvalues

        # Pronóstico: continuar el suavizamiento hacia adelante
        pronostico = modelo.forecast(steps=periodos)

        self.__entidad.guardar_resultados(suavizado)
        self.enviar_resultados(suavizado)
        return suavizado, pronostico

    def enviar_resultados(self, resultados):
        self.__resultados = resultados

    def enviar_mensaje(self, mensaje: str):
        self.__mensaje = mensaje

    def get_entidad(self):
        return self.__entidad


class CCAplicarArima:
    """Diagrama 4: CC-Aplicar Arima"""

    def __init__(self):
        self.__entidad = EArima()

    def validar_datos_y_parametros(self, p, d, q, periodos) -> dict:
        errores = []
        if p < 0 or d < 0 or q < 0:
            errores.append("p, d y q deben ser enteros no negativos.")
        if periodos < 1:
            errores.append("Los períodos deben ser al menos 1.")
        return {"valido": len(errores) == 0, "errores": errores}

    def calcular_arima(self, datos, p, d, q, periodos):
        validacion = self.validar_datos_y_parametros(p, d, q, periodos)
        if not validacion["valido"]:
            self.enviar_mensaje(" | ".join(validacion["errores"]))
            return None, None

        try:
            modelo    = SARIMAX(datos, order=(p, d, q),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            resultado  = modelo.fit(disp=False)
            ajuste     = resultado.fittedvalues
            pronostico = resultado.forecast(steps=periodos)

            self.__entidad.guardar_resultados(ajuste, pronostico)
            self.enviar_resultados(ajuste, pronostico)
            return ajuste, pronostico

        except Exception as e:
            self.enviar_mensaje(f"Error al ajustar el modelo: {e}")
            return None, None

    def enviar_resultados(self, ajuste, pronostico):
        self.__ajuste     = ajuste
        self.__pronostico = pronostico

    def enviar_mensaje(self, mensaje: str):
        self.__mensaje = mensaje

    def get_entidad(self):
        return self.__entidad


class CCVisualizarGraficos:
    """Diagrama 5: CC-Visualizar Gráficos"""

    def __init__(self, e_datos: EDatosHistoricos, e_ses: ESES, e_arima: EArima):
        self.__e_datos = e_datos
        self.__e_ses   = e_ses
        self.__e_arima = e_arima

    def solicitar_resultados(self) -> dict:
        return {
            "datos":     self.__e_datos.serie,
            "ses":       self.__e_ses.resultados,
            "ajuste":    self.__e_arima.ajuste,
            "pronostico": self.__e_arima.pronostico,
        }

    def generar_grafico(self, opcion: str):
        r = self.solicitar_resultados()

        if r["datos"] is None:
            self.enviar_mensaje("No hay datos cargados. Use la opción 3 primero.")
            return None

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(r["datos"].index, r["datos"], label="Serie original", linewidth=2)

        if opcion in ("ses", "ambos") and r["ses"] is not None:
            ax.plot(r["ses"].index, r["ses"], label="SES suavizado", linewidth=2)

        if opcion in ("arima", "ambos") and r["ajuste"] is not None:
            ax.plot(r["ajuste"].index, r["ajuste"], label="ARIMA ajustado", linewidth=2)

        if opcion in ("arima", "ambos") and r["pronostico"] is not None:
            ax.plot(r["pronostico"].index, r["pronostico"],
                    marker="o", label="Pronóstico ARIMA", linewidth=2)

        ax.set_title(f"SAIE — Visualización: {opcion.upper()}")
        ax.set_xlabel("Año")
        ax.set_ylabel("Valor")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig

    def enviar_mensaje(self, mensaje: str):
        print(f"\n  {mensaje}\n")

    def enviar_grafico(self, fig):
        if fig:
            plt.show()


class CCExportarGraficos:
    """Diagrama 6: CC-Exportar gráficos y resultados"""

    def solicitar_exportacion_con_formato(self, fig, pronostico, ruta_salida: str, formato: str):
        try:
            if formato == "imagen":
                ruta = ruta_salida + ".png"
                fig.savefig(ruta, dpi=150, bbox_inches="tight")
                self.enviar_mensaje(f"Gráfico guardado como imagen: {ruta}")

            elif formato == "pdf":
                from matplotlib.backends.backend_pdf import PdfPages
                ruta = ruta_salida + ".pdf"
                with PdfPages(ruta) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")
                self.enviar_mensaje(f"Gráfico guardado como PDF: {ruta}")

            elif formato == "excel":
                ruta = ruta_salida + ".xlsx"
                if pronostico is not None:
                    df_exp = pronostico.reset_index()
                    df_exp.columns = ["Fecha", "Pronóstico"]
                    df_exp.to_excel(ruta, index=False)
                    self.enviar_mensaje(f"Resultados exportados a Excel: {ruta}")
                else:
                    self.enviar_mensaje("No hay pronóstico ARIMA para exportar a Excel.")
            else:
                self.enviar_mensaje("Formato no reconocido.")

        except Exception as e:
            self.enviar_mensaje(f"Error al exportar: {e}")

    def enviar_mensaje(self, mensaje: str):
        print(f"\n  {mensaje}\n")


# =============================================================
# VISTAS
# =============================================================

LINEA  = "=" * 52
LINEA2 = "-" * 52


class UILogin:
    """Diagrama 1: UI-Login"""

    def __init__(self):
        self.__controlador = CCLogin()

    def ingresar_datos(self) -> None:
        self.__mostrar_encabezado()
        intentos     = 0
        max_intentos = 3

        while intentos < max_intentos:
            correo     = input("  Usuario (correo): ").strip()
            contrasena = input("  Contraseña:       ").strip()
            print()

            resultado = self.__controlador.validar_datos(correo, contrasena)

            if resultado["exitoso"]:
                print(f"  ✔  {resultado['mensaje']}\n")
                UIMenu(resultado["usuario"]).muestra_menu()
                return
            else:
                self.__controlador.enviar(f"✘  {resultado['mensaje']}")
                intentos  += 1
                restantes  = max_intentos - intentos
                if restantes > 0:
                    print(f"  Intentos restantes: {restantes}\n")

        print("\n  Número máximo de intentos alcanzado. Acceso bloqueado.\n")

    def mostrar_mensaje(self, mensaje: str):
        print(f"\n  {mensaje}\n")

    def __mostrar_encabezado(self):
        print()
        print(LINEA)
        print("                SAIE")
        print("     Sistema de Análisis de")
        print("       Indicadores Económicos")
        print(LINEA)
        print("     UI: LOGIN")
        print(LINEA)
        print()
        print("  Ingrese sus credenciales para continuar.")
        print()


class UIIngresarDatosHistoricos:
    """Diagrama 2: UI-Ingresar datos históricos"""

    def __init__(self):
        self.__controlador = CCIngresarDatosHistoricos()

    def ingresa_datos(self) -> EDatosHistoricos:
        print()
        print(LINEA)
        print("   SAIE — Ingresar datos históricos")
        print(LINEA)
        print()
        print("  Pegue la ruta de su archivo (Ctrl+V).")
        print("  Formatos soportados: .xlsx  .csv  .xml")
        print()

        ruta      = input("  Ruta: ").strip().strip('"')
        resultado = self.__controlador.validar_datos(ruta)

        self.mostrar_mensaje(resultado["mensaje"])

        if not resultado["exitoso"]:
            input("  Presione Enter para volver al menú...")
            return None

        try:
            # ── Formato Banco Mundial: elegir país ──
            if resultado.get("formato") == "banco_mundial":
                paises = resultado["paises"]
                print("  Países disponibles:")
                for i, p in enumerate(paises, start=1):
                    print(f"    [{i}] {p}")
                print()
                sel_pais = int(input("  Seleccione el número del país: ").strip()) - 1
                if sel_pais < 0 or sel_pais >= len(paises):
                    self.mostrar_mensaje("Selección no válida.")
                    input("  Presione Enter para volver al menú...")
                    return None
                pais    = paises[sel_pais]
                entidad = self.__controlador.cargar_desde_banco_mundial(
                    resultado["df"],
                    resultado["col_pais"],
                    resultado["cols_anios"],
                    pais,
                    ruta
                )
                self.mostrar_mensaje(entidad.mostrar_mensaje())
                input("  Presione Enter para volver al menú...")
                return entidad

            # ── Formato estándar: elegir columna ──
            else:
                columnas_num = resultado["columnas_num"]
                print("  Columnas disponibles:")
                for i, col in enumerate(columnas_num, start=1):
                    print(f"    [{i}] {col}")
                print()
                sel = int(input("  Seleccione el número de la columna: ").strip()) - 1
                if sel < 0 or sel >= len(columnas_num):
                    self.mostrar_mensaje("Selección no válida.")
                    input("  Presione Enter para volver al menú...")
                    return None
                columna = columnas_num[sel]
                entidad = self.__controlador.cargar_serie(
                    resultado["df"], resultado["col_anio"], columna, ruta
                )
                self.mostrar_mensaje(entidad.mostrar_mensaje())
                input("  Presione Enter para volver al menú...")
                return entidad

        except ValueError:
            self.mostrar_mensaje("Entrada inválida.")
            input("  Presione Enter para volver al menú...")
            return None

    def mostrar_mensaje(self, mensaje: str):
        print(f"\n  {mensaje}\n")


class UIAplicarSES:
    """Diagrama 3: UI-Aplicar SES"""

    def __init__(self):
        self.__controlador = CCAplicarSES()

    def selecciona_opcion_ses(self, datos) -> ESES:
        print()
        print(LINEA)
        print("   SAIE — Análisis SES")
        print(LINEA)
        print()

        try:
            alpha   = float(input("  Ingresa valor alfa (0 < α < 1, ej. 0.3): ").strip())
            periodos = int(input("  Ingresa período a pronosticar (ej. 5): ").strip())
        except ValueError:
            self.muestra_mensaje("Valores inválidos.")
            input("  Presione Enter para volver al menú...")
            return self.__controlador.get_entidad()

        suavizado, pronostico = self.__controlador.calcular_ses(datos, alpha, periodos)

        if suavizado is None:
            self.muestra_mensaje("No se pudo aplicar SES. Verifique el valor de α.")
            input("  Presione Enter para volver al menú...")
            return self.__controlador.get_entidad()

        self.muestra_resultados(datos, suavizado, pronostico)
        input("  Presione Enter para volver al menú...")
        return self.__controlador.get_entidad()

    def muestra_resultados(self, datos, suavizado, pronostico):
        self.muestra_mensaje("SES aplicado correctamente.")

        # Imprimir pronóstico en consola
        print("  Pronóstico SES:")
        for fecha, valor in pronostico.items():
            print(f"    {fecha.year}: {valor:.4f}")
        print()

        plt.figure(figsize=(11, 5))
        plt.plot(datos.index,     datos,     label="Serie original",  linewidth=2)
        plt.plot(suavizado.index, suavizado, label="Serie suavizada", linewidth=2)
        plt.plot(pronostico.index, pronostico,
                 marker="o", linestyle="--", label="Pronóstico SES", linewidth=2)
        plt.title("Suavizamiento Exponencial Simple — SAIE")
        plt.xlabel("Año")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def muestra_mensaje(self, mensaje: str):
        print(f"\n  {mensaje}\n")


class UIAplicarArima:
    """Diagrama 4: UI-Aplicar Arima"""

    def __init__(self):
        self.__controlador = CCAplicarArima()

    def selecciona_opcion_arima(self, datos) -> EArima:
        print()
        print(LINEA)
        print("   SAIE — Modelo ARIMA")
        print(LINEA)
        print()

        try:
            p       = int(input("  Ingresa p (ej. 1): ").strip())
            d       = int(input("  Ingresa d (ej. 1): ").strip())
            q       = int(input("  Ingresa q (ej. 1): ").strip())
            periodos = int(input("  Ingresa período a pronosticar (ej. 5): ").strip())
        except ValueError:
            self.muestra_mensaje("Valores inválidos.")
            input("  Presione Enter para volver al menú...")
            return self.__controlador.get_entidad()

        ajuste, pronostico = self.__controlador.calcular_arima(datos, p, d, q, periodos)

        if ajuste is None:
            self.muestra_mensaje("No se pudo aplicar ARIMA. Verifique los parámetros.")
            input("  Presione Enter para volver al menú...")
            return self.__controlador.get_entidad()

        self.muestra_resultado(datos, ajuste, pronostico)
        input("  Presione Enter para volver al menú...")
        return self.__controlador.get_entidad()

    def muestra_resultado(self, datos, ajuste, pronostico):
        self.muestra_mensaje("Modelo ARIMA aplicado correctamente.")
        plt.figure(figsize=(10, 5))
        plt.plot(datos.index,      datos,      label="Serie original", linewidth=2)
        plt.plot(ajuste.index,     ajuste,     label="ARIMA ajustado", linewidth=2)
        plt.plot(pronostico.index, pronostico, label="Pronóstico",
                 marker="o", linewidth=2)
        plt.title("Modelo ARIMA — SAIE")
        plt.xlabel("Año")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print("  Pronóstico:")
        for fecha, valor in pronostico.items():
            print(f"    {fecha.year}: {valor:.4f}")
        print()

    def muestra_mensaje(self, mensaje: str):
        print(f"\n  {mensaje}\n")


class UIVisualizarGraficos:
    """Diagrama 5: UI-Visualizar Gráficos"""

    def __init__(self, e_datos: EDatosHistoricos, e_ses: ESES, e_arima: EArima):
        self.__controlador = CCVisualizarGraficos(e_datos, e_ses, e_arima)
        self.__e_arima     = e_arima

    def selecciona_opcion(self):
        print()
        print(LINEA)
        print("   SAIE — Visualizar Gráficos")
        print(LINEA)
        print()
        print("  [1] Serie original + SES")
        print("  [2] Serie original + ARIMA")
        print("  [3] Todo junto")
        print()

        op = input("  Seleccione: ").strip()
        opciones = {"1": "ses", "2": "arima", "3": "ambos"}

        if op not in opciones:
            print("  Opción no válida.\n")
            input("  Presione Enter para volver al menú...")
            return

        fig = self.__controlador.generar_grafico(opciones[op])
        self.resultados(fig)
        self.muestra_grafico(fig)

        # Ofrecer exportación
        exportar = input("  ¿Desea exportar? (s/n): ").strip().lower()
        if exportar == "s":
            UIExportarGraficos(fig, self.__e_arima.pronostico).selecciona_opcion_exportar()
        else:
            input("  Presione Enter para volver al menú...")

    def resultados(self, fig):
        pass  # los resultados ya están en el gráfico

    def muestra_grafico(self, fig):
        if fig:
            plt.show()


class UIExportarGraficos:
    """Diagrama 6: UI-Exportar gráficos y resultados"""

    def __init__(self, fig, pronostico):
        self.__fig        = fig
        self.__pronostico = pronostico
        self.__controlador = CCExportarGraficos()

    def selecciona_opcion_exportar(self):
        print()
        print(LINEA)
        print("   SAIE — Exportar gráficos y resultados")
        print(LINEA)
        print()
        print("  Muestra: gráfico actual generado.")
        print()
        print("  Selecciona opción exportar:")
        print("  [1] PDF")
        print("  [2] Imagen (PNG)")
        print("  [3] Excel (pronóstico ARIMA)")
        print()

        op = input("  Seleccione formato: ").strip()
        formatos = {"1": "pdf", "2": "imagen", "3": "excel"}

        if op not in formatos:
            self.muestra_mensaje("Opción no válida.")
            input("  Presione Enter para volver al menú...")
            return

        formato = formatos[op]
        self.selecciona_opciones_de_formato(formato)

    def selecciona_opciones_de_formato(self, formato: str):
        ruta_salida = input("  Nombre del archivo de salida (sin extensión): ").strip()
        self.__controlador.solicitar_exportacion_con_formato(
            self.__fig, self.__pronostico, ruta_salida, formato
        )
        self.muestra_mensaje(f"Exportación completada en formato {formato.upper()}.")
        input("  Presione Enter para volver al menú...")

    def muestra_mensaje(self, mensaje: str):
        print(f"\n  {mensaje}\n")


class UIMenu:
    """Diagrama 1: UI-Menu"""

    def __init__(self, usuario: EUsuario):
        self.__usuario  = usuario
        self.__e_datos  = EDatosHistoricos()
        self.__e_ses    = ESES()
        self.__e_arima  = EArima()
        self.__fig      = None

    def muestra_menu(self) -> None:
        es_admin = self.__usuario.rol == "Administrador"
        while True:
            self.__mostrar_pantalla()
            opcion = input("  Seleccione una opción: ").strip()
            print()

            if opcion == "1":
                self.__inicio()

            elif opcion == "2" and es_admin:
                self.__control_acceso()

            elif (opcion == "2" and not es_admin) or (opcion == "3" and es_admin):
                # Ingresar datos: opción 2 analista, opción 3 admin
                ui = UIIngresarDatosHistoricos()
                entidad = ui.ingresa_datos()
                if entidad and entidad.serie is not None:
                    self.__e_datos = entidad

            elif (opcion == "3" and not es_admin) or (opcion == "4" and es_admin):
                # Análisis SES: opción 3 analista, opción 4 admin
                if self.__e_datos.serie is None:
                    print("  Primero debe ingresar los datos.\n")
                    input("  Presione Enter para volver al menú...")
                else:
                    ui = UIAplicarSES()
                    self.__e_ses = ui.selecciona_opcion_ses(self.__e_datos.serie)

            elif (opcion == "4" and not es_admin) or (opcion == "5" and es_admin):
                # Modelo ARIMA: opción 4 analista, opción 5 admin
                if self.__e_datos.serie is None:
                    print("  Primero debe ingresar los datos.\n")
                    input("  Presione Enter para volver al menú...")
                else:
                    ui = UIAplicarArima()
                    self.__e_arima = ui.selecciona_opcion_arima(self.__e_datos.serie)

            elif (opcion == "5" and not es_admin) or (opcion == "6" and es_admin):
                # Visualizar: opción 5 analista, opción 6 admin
                if self.__e_datos.serie is None:
                    print("  Primero debe ingresar los datos.\n")
                    input("  Presione Enter para volver al menú...")
                else:
                    ui = UIVisualizarGraficos(self.__e_datos, self.__e_ses, self.__e_arima)
                    ui.selecciona_opcion()

            elif opcion == "0":
                self.__cerrar_sesion()
                return

            else:
                print("  Opción no válida. Intente nuevamente.\n")
                input("  Presione Enter para continuar...")

    def muestra_mensaje(self, mensaje: str):
        print(f"\n  {mensaje}\n")

    def __mostrar_pantalla(self) -> None:
        estado = self.__e_datos.mostrar_mensaje()
        print()
        print(LINEA)
        print("   SAIE                    " + self.__usuario.rol)
        print(LINEA)
        print(f"  {estado}")
        print(LINEA2)
        print()
        if self.__usuario.rol == "Administrador":
            print("  ┌─────────────────────────────────────────────┐")
            print("  │  [1] Inicio                                 │")
            print("  │  [2] Control de acceso                      │")
            print("  │  [3] Ingresar datos históricos              │")
            print("  │  [4] Análisis SES                           │")
            print("  │  [5] Modelo ARIMA                           │")
            print("  │  [6] Visualizar gráficos                    │")
            print("  ├─────────────────────────────────────────────┤")
            print("  │  [0] Cerrar sesión                          │")
            print("  └─────────────────────────────────────────────┘")
        else:
            print("  ┌─────────────────────────────────────────────┐")
            print("  │  [1] Inicio                                 │")
            print("  │  [2] Ingresar datos históricos              │")
            print("  │  [3] Análisis SES                           │")
            print("  │  [4] Modelo ARIMA                           │")
            print("  │  [5] Visualizar gráficos                    │")
            print("  ├─────────────────────────────────────────────┤")
            print("  │  [0] Cerrar sesión                          │")
            print("  └─────────────────────────────────────────────┘")
        print()

    def __inicio(self) -> None:
        print()
        print(LINEA)
        print("   SAIE — Inicio")
        print(LINEA)
        print(f"  Bienvenido, {self.__usuario.rol}")
        print()
        print("  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐")
        print("  │  Ingresar datos  │  │  Análisis SES    │  │  Modelo ARIMA    │")
        print("  └──────────────────┘  └──────────────────┘  └──────────────────┘")
        print()
        print("  ┌──────────────────────────────────────────┐")
        print("  │  Visualizar gráficos y resultados        │")
        print("  └──────────────────────────────────────────┘")
        print()
        input("  Presione Enter para volver al menú...")

    def __control_acceso(self) -> None:
        print()
        print(LINEA)
        print("   SAIE — Control de acceso")
        print(LINEA)
        print(f"  Usuario activo: {self.__usuario.usuario}")
        print(f"  Rol:            {self.__usuario.rol}")
        print()
        input("  Presione Enter para volver al menú...")

    def __cerrar_sesion(self) -> None:
        print()
        print("  Sesión cerrada. Hasta pronto.")
        print()


# =============================================================
# PUNTO DE ENTRADA
# =============================================================

if __name__ == "__main__":
    ui = UILogin()
    ui.ingresar_datos()