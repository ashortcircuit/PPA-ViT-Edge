import os
import shutil
import pandas as pd

# ===================== CONFIGURACIÓN =====================
CSV_FOLDS = "lista_folds_dataset.csv"
DIR_IMAGENES_UNICAS = "Dataset"  # Carpeta extraída de Curated_Preprocessed_Datset.zip
DIR_SALIDA_FOLDS = "Dataset_3Folds"

def reconstruir_folds():
    if not os.path.exists(CSV_FOLDS):
        print(f"❌ No se encontró el archivo CSV: {CSV_FOLDS}")
        return
    if not os.path.exists(DIR_IMAGENES_UNICAS):
        print(f"❌ No se encontró la carpeta '{DIR_IMAGENES_UNICAS}'. Extrae Curated_Preprocessed_Datset.zip primero.")
        return

    df = pd.read_csv(CSV_FOLDS)
    copiadas = 0
    
    for _, row in df.iterrows():
        archivo = str(row["Archivo"]).strip()
        clase = str(row["Clase"]).strip()
        
        # Buscar la imagen original o con el sufijo _2_exclusion
        ruta_origen = os.path.join(DIR_IMAGENES_UNICAS, clase, archivo)
        if not os.path.exists(ruta_origen):
            nombre, ext = os.path.splitext(archivo)
            ruta_origen = os.path.join(DIR_IMAGENES_UNICAS, clase, f"{nombre}_2_exclusion{ext}")
            
        if not os.path.exists(ruta_origen):
            print(f"⚠️ No se encontró: {archivo} en {os.path.join(DIR_IMAGENES_UNICAS, clase)}")
            continue
            
        # Copiar a cada Fold según indique el CSV (Train o Test)
        for fold in ["Fold_1", "Fold_2", "Fold_3"]:
            if fold in df.columns:
                split = str(row[fold]).strip()
                if split in ["Train", "Test"]:
                    dir_destino = os.path.join(DIR_SALIDA_FOLDS, fold, split, clase)
                    os.makedirs(dir_destino, exist_ok=True)
                    shutil.copy2(ruta_origen, os.path.join(dir_destino, os.path.basename(ruta_origen)))
        copiadas += 1

    print(f"✅ Estructura de 3 Folds reconstruida con éxito en: '{DIR_SALIDA_FOLDS}' ({copiadas} imágenes base procesadas).")

if __name__ == "__main__":
    reconstruir_folds()