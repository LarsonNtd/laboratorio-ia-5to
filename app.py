
import json
import os
import shutil
from pathlib import Path
import numpy as np
import gradio as gr
import onnxruntime as ort

from tokenizers import Tokenizer


carpeta = str(Path(__file__).resolve().parent / "modelo")
archivo_modelo = Path(carpeta) / "modelo.onnx"

def preparar_modelo():
    if archivo_modelo.exists():
        return

    informacion = json.loads(
        (Path(carpeta) / "modelo_info.json").read_text(
            encoding="utf-8"
        )
    )

    partes = sorted(Path(carpeta).glob("modelo.parte*"))

    if len(partes) != informacion["partes"]:
        raise RuntimeError("Faltan partes del modelo.")

    temporal = archivo_modelo.with_suffix(".tmp")

    with open(temporal, "wb") as destino:
        for parte in partes:
            with open(parte, "rb") as fuente:
                shutil.copyfileobj(
                    fuente, destino, length=1024 * 1024
                )

    if temporal.stat().st_size != informacion["bytes"]:
        raise RuntimeError("El modelo quedó incompleto.")

    temporal.replace(archivo_modelo)

preparar_modelo()


tokenizador = Tokenizer.from_file(
    f"{carpeta}/tokenizer.json"
)
tokenizador.enable_truncation(max_length=128)

with open(
    f"{carpeta}/etiquetas.json",
    encoding="utf-8"
) as archivo:
    etiquetas = json.load(archivo)

opciones = ort.SessionOptions()
opciones.intra_op_num_threads = 1
opciones.inter_op_num_threads = 1
opciones.enable_cpu_mem_arena = False
opciones.enable_mem_pattern = False

sesion = ort.InferenceSession(
    f"{carpeta}/modelo.onnx",
    sess_options=opciones,
    providers=["CPUExecutionProvider"]
)

def analizar_sentimiento(texto):
    if not texto or not texto.strip():
        return "Escribe una frase para analizar."

    tokens = tokenizador.encode(texto)

    entradas = {
        "input_ids": np.array(
            [tokens.ids], dtype=np.int64
        ),
        "attention_mask": np.array(
            [tokens.attention_mask], dtype=np.int64
        )
    }

    logits = sesion.run(None, entradas)[0][0]
    valores = np.exp(logits - np.max(logits))
    probabilidades = valores / valores.sum()
    indice = int(np.argmax(probabilidades))

    traducciones = {
        "negative": "Negativo",
        "neutral": "Neutral",
        "positive": "Positivo"
    }

    etiqueta = etiquetas[str(indice)]
    sentimiento = traducciones.get(
        etiqueta.lower(), etiqueta
    )
    puntuacion = probabilidades[indice] * 100

    return (
        f"Sentimiento detectado: {sentimiento}\n"
        f"Puntuación del modelo: {puntuacion:.1f}%"
    )



demo_sentimientos = gr.Interface(
    fn=analizar_sentimiento,
    inputs=gr.Textbox(
        label="Escribe una frase",
        placeholder="Ejemplo: Me encantó esta película."
    ),
    outputs=gr.Textbox(label="Resultado del análisis"),
    title="Analizador de sentimientos con IA",
    description=(
        "Escribe una frase y el modelo intentará "
        "identificar su sentimiento."
    ),
    examples=[
        ["Me encantó esta película, fue muy divertida."],
        ["El servicio fue terrible y estoy decepcionado."],
        ["Hoy fui al supermercado a comprar alimentos."]
    ]
)

if __name__ == "__main__":
    demo_sentimientos.queue(max_size=10)
    demo_sentimientos.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        share=False
    )
