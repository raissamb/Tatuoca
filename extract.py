# from numpy import (ndarray, dtype, int64,
#   float64, zeros, vstack, errstate, arange)
# from os.path import isfile, isdir, join, splitext
# from os import mkdir, scandir
# from itertools import pairwise, product
# from cv2.typing import MatLike, Size
# from cv2 import (imread, imwrite, resize, Canny, cvtColor,
#   GaussianBlur, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
#   adaptiveThreshold, COLOR_BGR2GRAY, COLOR_GRAY2BGR)
# from lpmalgos import Ellipsoid, find_clusters
# from skeleton import skeletonize
# from shutil import rmtree


# Imports using pathlib
from pathlib import Path
import json
from itertools import pairwise, product
from numpy import (ndarray, dtype, int64, float64, zeros, vstack, errstate, arange)
from cv2.typing import MatLike, Size
from cv2 import (imread, imwrite, resize, Canny, cvtColor,
                 GaussianBlur, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
                 adaptiveThreshold, COLOR_BGR2GRAY, COLOR_GRAY2BGR)
from lpmalgos import Ellipsoid, find_clusters
from skeleton import skeletonize
from shutil import rmtree
import numpy as np
from datetime import datetime, timedelta
import re
import cv2
import pandas as pd




# Definitions
NDArray = ndarray[tuple[int, int], dtype[int64 | float64]]
BLACK, WHITE, RED, GREEN = 0, 255, (0, 0, 255), (0, 255, 0)
NDArrays = list[NDArray]


#### Functions

def bgr2gray(image: MatLike) -> MatLike:
  """
  Converte uma imagem de BGR para escala de cinza.
  
  Args:
    image (MatLike): Imagem em formato BGR.
  
  Returns:
    MatLike: Imagem em escala de cinza.
  """
  #return cvtColor(image, COLOR_BGR2GRAY)
  
  if len(image.shape) == 2:
      return image
  return cvtColor(image, COLOR_BGR2GRAY)



def gray2bgr(image: MatLike) -> MatLike:
  """
  Converte uma imagem de escala de cinza para BGR.
  
  Args:
    image (MatLike): Imagem em escala de cinza.
  
  Returns:
    MatLike: Imagem em formato BGR.
  """
  return cvtColor(image, COLOR_GRAY2BGR)



### Class
class ImageOpener:
  """
  Classe para abrir imagens de um diretório ou arquivo.
  """
  __slots__ = ("path", "index")

  def __init__(self, path: str):
    """
    Inicializa a classe com o caminho do arquivo ou diretório.
    
    Args:
      path (str): Caminho do arquivo ou diretório.
    
    Raises:
      FileNotFoundError: Se o caminho não existir.
    """

    self.path = Path(path)
    if not self.path.exists():
        raise FileNotFoundError(f"{str(self.path)!r} não existe!")
    self.index = None

  
  def __getitem__(self, index: int | slice):
    """
    Define o índice ou fatia para iteração.
    
    Args:
      index (int | slice): Índice ou fatia.
    
    Returns:
      self: Instância da classe.
    """
    if isinstance(index, (int, slice)):
      self.index = index
    return self
  

  def __iter__(self):
    """
    Itera sobre os arquivos de imagem no diretório ou arquivo.
    
    Yields:
      tuple: Nome do arquivo e imagem em escala de cinza.
    """

    valid_extensions = {
            ".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png",
            ".pnm", ".avif", ".pbm", ".pgm", ".ppm", ".pxm", ".webp",
            ".pfm", ".sr", ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic"
        }

    def open_image_core(file_path: Path):
      if file_path.suffix.lower() not in valid_extensions:
          raise TypeError(f"{str(file_path)!r} não é uma imagem suportada!")
            
      image = imread(str(file_path), 0)
      if image is None:
          raise TypeError(f"{str(file_path)!r} não pôde ser aberto!")
      
      # Diretório de saída com o mesmo nome base da imagem (sem extensão)
      output_dir = file_path.parent / file_path.stem
      if output_dir.is_dir():
          rmtree(output_dir)
      output_dir.mkdir(parents=True, exist_ok=True)
      
      return output_dir, bgr2gray(image)

    if self.path.is_file():
        yield open_image_core(self.path)
        return

    image_files = sorted([p for p in self.path.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions])

    if isinstance(self.index, int):
        yield open_image_core(image_files[self.index])
    elif isinstance(self.index, slice):
        yield from (open_image_core(p) for p in image_files[self.index])
    else:
        yield from (open_image_core(p) for p in image_files)

    
    
def cut_image(image: MatLike, shape: Size, output_dir: Path | None = None) -> MatLike:
  """
  Corta a imagem em uma região específica e redimensiona.

  Args:
    image (MatLike): Imagem de entrada.
    shape (Size): Tamanho desejado da imagem.

  Raises:
    TypeError: Se o shape não tiver tamanho 2.

  Returns:
    MatLike: Imagem cortada e redimensionada.
  """
  if len(shape) != 2:
    raise TypeError("shape sem tamanho 2!")

  h_orig, w_orig = image.shape[:2]

  # Cálculo da bounding box
  y_min, y_max = int(h_orig * 0.15), int(h_orig * 0.92)
  x_min, x_max = int(w_orig * 0.05), int(w_orig * 0.95)

  cropped = image[y_min:y_max, x_min:x_max]
  h_crop, w_crop = cropped.shape[:2]

  resized = resize(cropped, shape[::-1])

  if output_dir is not None:
      metadata = {
          "original_shape": {"height_px": h_orig, "width_px": w_orig},
          "crop_box": {"y_min": y_min, "y_max": y_max, "x_min": x_min, "x_max": x_max},
          "cropped_shape": {"height_px": h_crop, "width_px": w_crop},
          "target_shape": {"height_px": shape[0], "width_px": shape[1]},
          "scale_ratios": {
              "ratio_y": round(h_crop / shape[0], 4),
              "ratio_x": round(w_crop / shape[1], 4)
          }
      }
      meta_file = output_dir / "crop_metadata.json"
      with meta_file.open("w", encoding="utf-8") as f:
          json.dump(metadata, f, indent=2)

  print("cut_image terminado!")
  return resized


def gauss_process(image: MatLike, size: int = 19, C: int = 10) -> MatLike:
  """
  Aplica processamento Gaussiano na imagem.

  Args:
    image (MatLike): Imagem de entrada.
    size (int, optional): Tamanho do bloco. Padrão é 19.
    C (int, optional): Constante subtraída da média. Padrão é 10.

  Returns:
    MatLike: Imagem processada.
  """
  output = adaptiveThreshold(image, 255,
    ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, size, C)
  output = GaussianBlur(output, (11, 11), 0)
  print("gauss_process terminado!")
  return output



def canny_filter(image: MatLike) -> MatLike:
  """
  Aplica filtro Canny na imagem.

  Args:
    image (MatLike): Imagem de entrada.

  Returns:
    MatLike: Imagem filtrada.
  """
  output = Canny(image, 200, 255, L2gradient = True)
  print("canny_filter terminado!")
  return output



def dbscan_separation(image: MatLike, min_points: int = 2000) -> NDArrays:
  """
  Separa clusters usando DBSCAN.

  Args:
    image (MatLike): Imagem de entrada.
    min_points (int, optional): Número mínimo de pontos em um cluster. Padrão é 2000.

  Returns:
    NDArrays: Lista de clusters separados.
  """
  def dbscan_separation_core():
    dim1, dim2 = (image == WHITE).nonzero()
    locs = zeros((len(dim1), 3), int)
    locs.T[0], locs.T[1] = dim2, dim1
    ani = Ellipsoid(((.3, 0, 0), (0, 2.6, 0), (0, 0, 0)))
    clusters = find_clusters(locs, ani, 20, 6, 1e-9, 1)
    for cluster in set(clusters):
      dim2, dim1, _ = locs[clusters == cluster].T
      if len(dim2) < min_points: continue
      yield vstack((dim1, dim2)).T[dim2.argsort()]

  output = list(dbscan_separation_core())
  print("dbscan_separation terminado!")
  return output


def curve_adjust(D: NDArray, I: NDArray, degree: int) -> NDArray:
  """
  Ajusta curvas usando um polinômio de grau n.

  Args:
    D (NDArray): Domínio dos valores de entrada.
    I (NDArray): Imagem dos valores de entrada.
    degree (int): Grau do polinômio.

  Returns:
    NDArray: Curva ajustada.
  """
  D, I = D.astype(float), I.astype(float)
  with errstate(all = "raise"):
    G = D.reshape(-1, 1) ** range(degree)
    A, B = (G.T @ G), (G.T @ I)
    L, U = zeros((degree, degree)), A
    L.flat[::(degree + 1)] = 1
    for i, ii in pairwise(range(degree)):
      L[ii:, i] = U[ii:, i] / U[i, i]
      U[ii:] -= L[ii:, i, None] * U[i]
    x, y = zeros(degree), zeros(degree)
    for i in range(degree):
      y[i] = B[i] - (L[i] @ y)
    for i in reversed(range(degree)):
      temp = y[i] - (U[i] @ x)
      x[i] = temp / U[i, i]
    return (G @ x)


def line_separation(clusters: NDArrays) -> tuple[NDArrays, NDArrays]:
  """
  Separa linhas em curvas e bases.

  Args:
    clusters (NDArrays): Lista de clusters.

  Returns:
    tuple[NDArrays, NDArrays]: Curvas e bases separadas.
  """
  MIN_POINTS, MAX_DIFF = 550, 6
  curves: NDArrays = []
  baselines: NDArrays = []
  for cluster in clusters:
    (A, B) = cluster.T
    choice = curves
    line = curve_adjust(B, A, 2)
    res = (abs(line - A) < MAX_DIFF)
    if (~res).sum() < MIN_POINTS:
      choice = baselines
      A, B = A[res], B[res]
    cluster = vstack((A, B)).T
    choice.append(cluster)
  print("line_separation terminado!")
  return curves, baselines


def save_lines(lines: NDArrays, output_dir: Path, dtype: str, shape: Size):
  """
  Salva linhas em arquivos de imagem e texto.

  Args:
    lines (NDArrays): Lista de linhas.
    directory (str): Diretório para salvar os arquivos.
    dtype (str): Tipo de linha (curva ou base).
    shape (Size): Tamanho da imagem.

  Raises:
    TypeError: Se o shape não tiver tamanho 2.
  """
  if len(shape) != 2:
        raise TypeError("shape sem tamanho 2!")
  lines_dir = output_dir / "lines"
  lines_dir.mkdir(parents=True, exist_ok=True)
  black_image = zeros(shape, "uint8")

  for i, subarray in enumerate(lines):
      image, (dim1, dim2) = black_image.copy(), subarray.T
      image[dim1, dim2] = WHITE
      imwrite(str(lines_dir / f"{dtype}{i}.png"), image)
      
      with (lines_dir / f"{dtype}{i}.txt").open("w", encoding="utf-8") as out:
          out.write(f"Tamanho da imagem: {shape}\n")
          out.writelines(f"{elem}\n" for elem in subarray)

  print("save_lines terminado!")
  

def overlay_lines(image: MatLike, lines: NDArrays) -> MatLike:
  """
  Sobrepõe linhas em uma imagem.

  Args:
    image (MatLike): Imagem de entrada.
    lines (NDArrays): Lista de linhas.

  Returns:
    MatLike: Imagem com linhas sobrepostas.
  """
  output = gray2bgr(image)
  dim1, dim2 = zip(*(e for line in lines for e in line))
  output[dim1, dim2] = RED
  print("overlay_lines terminado!")
  return output


def calculate_differences(curves: NDArrays, baselines: NDArrays, 
                          output_dir: Path, px_to_mm: float):  
  """
  Calcula diferenças entre curvas e bases.

  Args:
    curves (NDArrays): Lista de curvas.
    baselines (NDArrays): Lista de bases.
    directory (str): Diretório para salvar os arquivos.
    px_to_mm (float): Constante de conversão.
  """

  differences_dir = output_dir / "diffs"
  differences_dir.mkdir(parents=True, exist_ok=True)
  curves_and_baselines = product(enumerate(curves), enumerate(baselines))

  for (curve_idx, curve), (base_idx, baseline) in curves_and_baselines:
      mean = baseline.T[0].mean(None, int)
      values = ((mean - curve.T[0]) * px_to_mm).round(1)
      csv_file = differences_dir / f"curve{curve_idx}base{base_idx}.csv"

      with csv_file.open("w", encoding="utf-8") as out:
          out.write("row_trace,row_baseline_mean,column,distance_mm\n")
          for (r_trace, col), dist in zip(curve, values):
              out.write(f"{r_trace},{mean},{col},{dist:.1f}\n")

  print("calculate_differences terminado!")
  

def get_30min_values(image: MatLike, curves: NDArrays, 
                     baselines: NDArrays, output_dir: Path, px_to_mm: float):
  """
  Obtém valores de 30 minutos.

  Args:
    curves (NDArrays): Lista de curvas.
    baselines (NDArrays): Lista de bases.
    directory (str): Diretório para salvar os arquivos.
    px_to_mm (float): Constante de conversão.

  Returns:
    tuple: Valores h, d, z de 30 em 30 minutos.
  """
  output = gray2bgr(image)
  dir_30min = output_dir / "30min"
  dir_30min.mkdir(parents=True, exist_ok=True)

  for i, (curve, baseline) in enumerate(product(curves, baselines)):
      ybase, xbase = baseline.T
      ymean = ybase.mean().round().astype(int64)
      xpass = (xbase[-1] - xbase[0]) / 48
      xvalues = ((arange(49) * xpass) + xbase[0]).round().astype(int64)
      indexes = (curve[:, 1, None] == xvalues).any(1)
      yvalues = ((ymean - curve[indexes, 0]) * px_to_mm).round(1)

      with (dir_30min / f"30min{i}.csv").open("w", encoding="utf-8") as outfile:
          outfile.write("column,distance_mm\n")
          outfile.writelines(f"{xv},{yv:.1f}\n" for xv, yv in zip(xvalues, yvalues))

      output[:, xvalues] = GREEN

  print("get_30min_values terminado!")
  return output


def magnetogram_image_scale(scale: int):
  """
  Calcula o tamanho da imagem e o fator de escala.
  Args:
    scale (int): Fator de escala para a imagem.
  Returns:
    tuple: Tamanho da imagem (linhas, colunas) e fator de escala.
  """
  if not isinstance(scale, int) or scale <= 0:
    raise TypeError("scale deve ser um inteiro positivo!")
  # O magnetograma tem 200x510 mm
  lines = 200 * scale
  columns = 510 * scale
  px_to_mm = 1 / scale
  return (lines, columns), px_to_mm


###

### The compute time marks produced errors when creating the time markings
# def compute_time_marks(
#     baseline_coords: np.ndarray,
#     output_dir: Path,
#     approach: int = 3,
#     start_date: str = "1964-09-12",
#     start_period: str = "morning",
#     gap2_col: float = 995.0,
#     second_to_last_gap_col: float = 4312.0,
#     gap2_k: int = 2,
#     second_to_last_k: int = 17,
#     gap1_col: float = 776.5,
#     gap1_k: int = 1
# ):
#     """
#     Computes baseline hourly time marks using Approach 2 or Approach 3.

#     Args:
#         baseline_coords (np.ndarray): Array of [row, col] coordinates from baselines[0].
#         output_dir (Path): Output directory for the magnetogram.
#         approach (int): 2 for adjacent-gap rate, 3 for long-baseline span rate.
#         start_date (str): Date string 'YYYY-MM-DD' representing the start date on paper.
#         start_period (str): 'morning' (default, nominal start at 12 UT / 09h local)
#                             or 'night' (nominal start at 00 UT / 21h local).
#         gap2_col (float): Column of 2nd gap (start of gap or gap center).
#         second_to_last_gap_col (float): Column of second-to-last gap.
#         gap2_k (int): Hour index of 2nd gap relative to nominal start (default k=2).
#         second_to_last_k (int): Hour index of second-to-last gap (default k=17).
#         gap1_col (float): Column of 1st gap (used for Approach 2).
#         gap1_k (int): Hour index of 1st gap (default k=1).
#     """
#     lines_dir = output_dir / "lines"
#     lines_dir.mkdir(parents=True, exist_ok=True)

#     # Sort baseline by column
#     sorted_baseline = baseline_coords[np.argsort(baseline_coords[:, 1])]
#     cols = sorted_baseline[:, 1]
#     rows = sorted_baseline[:, 0]

#     first_col, first_row = int(cols[0]), int(rows[0])
#     last_col, last_row = int(cols[-1]), int(rows[-1])

#     # Determine nominal start hour
#     nominal_start_utc = 12 if start_period.lower() == "morning" else 0
#     base_datetime = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(hours=nominal_start_utc)

#     # Calculate px_per_hour based on the chosen approach
#     if approach == 2:
#         # Rate from Gap 1 (k=1) to Gap 2 (k=2)
#         px_per_hour = (gap2_col - gap1_col) / (gap2_k - gap1_k)
#         anchor_col = gap1_col
#         anchor_dt = base_datetime + timedelta(hours=gap1_k)
#     elif approach == 3:
#         # Rate across long baseline span: from Gap 2 to Second-to-last Gap
#         px_per_hour = (second_to_last_gap_col - gap2_col) / (second_to_last_k - gap2_k)
#         anchor_col = gap2_col
#         anchor_dt = base_datetime + timedelta(hours=gap2_k)
#     else:
#         raise ValueError("Approach must be 2 or 3.")

#     # Helper function to get row coordinate for a target column
#     def get_matched_row(target_c):
#         idx = np.argmin(np.abs(cols - target_c))
#         return int(rows[idx])

#     # Calculate timestamps for first and last pixel
#     first_dt_utc = anchor_dt - timedelta(hours=(anchor_col - first_col) / px_per_hour)
#     last_dt_utc = anchor_dt + timedelta(hours=(last_col - anchor_col) / px_per_hour)

#     prefix = f"approach{approach}"
#     csv_file = lines_dir / f"{prefix}_basel0_time_marks.csv"

#     with csv_file.open("w", encoding="utf-8") as f:
#         f.write(
#             "date_utc,time_utc,date_local_utc3,time_local_utc3,"
#             "row,column,marker_type,px_per_hour_rate\n"
#         )

#         def write_record(dt_utc, r, c, mtype):
#             dt_loc = dt_utc - timedelta(hours=3)
#             f.write(
#                 f"{dt_utc.strftime('%Y-%m-%d')},{dt_utc.strftime('%H:%M:%S')},"
#                 f"{dt_loc.strftime('%Y-%m-%d')},{dt_loc.strftime('%H:%M:%S')},"
#                 f"{r},{c},{mtype},{px_per_hour_rate:.2f}\n"
#             )

#         px_per_hour_rate = px_per_hour

#         # 1. First Pixel
#         write_record(first_dt_utc, first_row, first_col, "first_pixel")

#         # 2. Hourly time marks across the record (24 hours span)
#         for k in range(1, 24):
#             target_col = anchor_col + (k - (gap1_k if approach == 2 else gap2_k)) * px_per_hour
#             if target_col < first_col or target_col > last_col:
#                 continue

#             matched_row = get_matched_row(target_col)
#             mark_dt = base_datetime + timedelta(hours=k)
#             write_record(mark_dt, matched_row, int(round(target_col)), "hourly_mark")

#         # 3. Last Pixel
#         write_record(last_dt_utc, last_row, last_col, "last_pixel")

#     print(f"[{prefix.upper()}] Time marks saved to: {csv_file}")



def detect_baseline_gaps(baseline_coords: np.ndarray, min_gap_width: int = 2) -> list[float]:
    """
    Automatically detects column centers of time-mark gaps directly from baseline coordinates.
    """
    # Extract unique sorted columns present in the baseline
    unique_cols = np.sort(np.unique(baseline_coords[:, 1]))
    diffs = np.diff(unique_cols)

    # Locate missing column intervals
    gap_indices = np.where(diffs >= min_gap_width)[0]
    detected_gaps = []

    for idx in gap_indices:
        col_before = unique_cols[idx]
        col_after = unique_cols[idx + 1]
        gap_center = (col_before + col_after) / 2.0
        detected_gaps.append(gap_center)

    return detected_gaps


# Version 2: using the min gap width examples
# def compute_time_marks(
#     baseline_coords: np.ndarray,
#     output_dir: Path,
#     approach: int = 3,
#     start_date: str = "1964-09-12",
#     start_period: str = "morning"
# ):
#     """
#     Dynamically identifies baseline gaps and computes hourly marks without hardcoded coordinates.
#     """
#     lines_dir = output_dir / "lines"
#     lines_dir.mkdir(parents=True, exist_ok=True)

#     sorted_baseline = baseline_coords[np.argsort(baseline_coords[:, 1])]
#     cols = sorted_baseline[:, 1]
#     rows = sorted_baseline[:, 0]

#     first_col, first_row = int(cols[0]), int(rows[0])
#     last_col, last_row = int(cols[-1]), int(rows[-1])

#     # 1. Automatically find all physical gaps along the baseline
#     gaps = detect_baseline_gaps(baseline_coords, min_gap_width=2)
    
#     if len(gaps) < 2:
#         raise ValueError(f"Fewer than 2 gaps detected along baseline in {output_dir.name}.")

#     # Gap 0 is the 1st gap after start (13h UT)
#     # Gap 1 is the 2nd gap (14h UT)
#     gap1_col = gaps[0]
#     gap2_col = gaps[1]
#     second_to_last_gap_col = gaps[-2] if len(gaps) >= 4 else gaps[-1]
    
#     # 2. Determine time-rate based on chosen approach
#     nominal_start_utc = 12 if start_period.lower() == "morning" else 0
#     base_datetime = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(hours=nominal_start_utc)

#     if approach == 2:
#         # Rate between first gap (13h UT, k=1) and second gap (14h UT, k=2)
#         px_per_hour = (gap2_col - gap1_col) / 1.0
#         anchor_col = gap1_col
#         anchor_dt = base_datetime + timedelta(hours=1)
#     elif approach == 3:
#         # Rate across long span from Gap 2 to second-to-last gap
#         # Estimate number of elapsed hours based on nominal variometer drum speed (~180-220 px/h)
#         span_pixels = second_to_last_gap_col - gap2_col
#         elapsed_hours = max(1, int(round(span_pixels / 180.0)))
        
#         px_per_hour = span_pixels / elapsed_hours
#         anchor_col = gap2_col
#         anchor_dt = base_datetime + timedelta(hours=2)
#     else:
#         raise ValueError("Approach must be 2 or 3.")

#     def get_matched_row(target_c):
#         idx = np.argmin(np.abs(cols - target_c))
#         return int(rows[idx])

#     first_dt_utc = anchor_dt - timedelta(hours=(anchor_col - first_col) / px_per_hour)
#     last_dt_utc = anchor_dt + timedelta(hours=(last_col - anchor_col) / px_per_hour)

#     prefix = f"approach{approach}"
#     csv_file = lines_dir / f"{prefix}_basel0_time_marks.csv"

#     with csv_file.open("w", encoding="utf-8") as f:
#         f.write(
#             "date_utc,time_utc,date_local_utc3,time_local_utc3,"
#             "row,column,marker_type,px_per_hour_rate\n"
#         )

#         def write_record(dt_utc, r, c, mtype):
#             dt_loc = dt_utc - timedelta(hours=3)
#             f.write(
#                 f"{dt_utc.strftime('%Y-%m-%d')},{dt_utc.strftime('%H:%M:%S')},"
#                 f"{dt_loc.strftime('%Y-%m-%d')},{dt_loc.strftime('%H:%M:%S')},"
#                 f"{r},{c},{mtype},{px_per_hour:.2f}\n"
#             )

#         # First pixel
#         write_record(first_dt_utc, first_row, first_col, "first_pixel")

#         # Hourly marks across 24h
#         for k in range(1, 25):
#             target_col = anchor_col + (k - (1 if approach == 2 else 2)) * px_per_hour
#             if target_col < first_col or target_col > last_col:
#                 continue

#             matched_row = get_matched_row(target_col)
#             mark_dt = base_datetime + timedelta(hours=k)
#             write_record(mark_dt, matched_row, int(round(target_col)), "hourly_mark")

#         # Last pixel
#         write_record(last_dt_utc, last_row, last_col, "last_pixel")

#     print(f"[{prefix.upper()}] Time marks auto-calibrated ({px_per_hour:.2f} px/h): {csv_file}")


def compute_time_marks_robust(
    baseline_coords: np.ndarray,
    output_dir: Path,
    start_date: str = "1964-09-12",
    start_period: str = "morning",
    search_window_px: int = 14
):
    """
    Robust 24-hour baseline calibration:
    1. Constrains total baseline length to 24.0h (Zero Cumulative Drift).
    2. Uses guided local search around each expected hour to snap to physical gaps.
    """
    lines_dir = output_dir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)

    sorted_baseline = baseline_coords[np.argsort(baseline_coords[:, 1])]
    cols = sorted_baseline[:, 1]
    rows = sorted_baseline[:, 0]

    first_col, first_row = int(cols[0]), int(rows[0])
    last_col, last_row = int(cols[-1]), int(rows[-1])

    # 1. Global Boundary Closure (Exact 24h baseline span)
    total_span_px = last_col - first_col
    global_rate = total_span_px / 24.0

    nominal_start_utc = 12 if start_period.lower() == "morning" else 0
    base_datetime = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(hours=nominal_start_utc)

    # Fast column lookup to detect physical gaps in the baseline
    unique_cols = set(cols)

    def get_matched_row(target_c):
        idx = np.argmin(np.abs(cols - target_c))
        return int(rows[idx])

    # 2. Guided Local Snapping for each hour (k = 0 to 24)
    calibrated_marks = []
    
    for k in range(25):
        expected_c = first_col + k * global_rate
        
        # Guided search in [expected_c - search_window_px, expected_c + search_window_px]
        w_start = int(round(expected_c - search_window_px))
        w_end = int(round(expected_c + search_window_px))
        
        # Check which columns in this window are missing from the baseline
        missing_cols = [c for c in range(w_start, w_end + 1) if c not in unique_cols]
        
        # If there are contiguous missing columns (a true physical gap), snap to its center
        if len(missing_cols) >= 2:
            # Group into contiguous missing segments and find the closest to expected_c
            gaps = []
            current_gap = [missing_cols[0]]
            for c in missing_cols[1:]:
                if c == current_gap[-1] + 1:
                    current_gap.append(c)
                else:
                    gaps.append(current_gap)
                    current_gap = [c]
            gaps.append(current_gap)
            
            # Select gap closest to expected position
            best_gap = min(gaps, key=lambda g: abs(np.mean(g) - expected_c))
            snapped_c = int(round(np.mean(best_gap)))
            marker_type = "snapped_gap"
        else:
            snapped_c = int(round(expected_c))
            marker_type = "interpolated"

        # Special labels for endpoints
        if k == 0:
            snapped_c = first_col
            marker_type = "first_pixel"
        elif k == 24:
            snapped_c = last_col
            marker_type = "last_pixel"

        matched_r = get_matched_row(snapped_c)
        mark_dt_utc = base_datetime + timedelta(hours=k)
        mark_dt_loc = mark_dt_utc - timedelta(hours=3)

        calibrated_marks.append({
            "date_utc": mark_dt_utc.strftime("%Y-%m-%d"),
            "time_utc": mark_dt_utc.strftime("%H:%M:%S"),
            "date_local_utc3": mark_dt_loc.strftime("%Y-%m-%d"),
            "time_local_utc3": mark_dt_loc.strftime("%H:%M:%S"),
            "row": matched_r,
            "column": snapped_c,
            "marker_type": marker_type,
            "px_per_hour_rate": round(global_rate, 2)
        })

    # Save to approach3 CSV file
    csv_file = lines_dir / "approach3_basel0_time_marks.csv"
    with csv_file.open("w", encoding="utf-8") as f:
        f.write("date_utc,time_utc,date_local_utc3,time_local_utc3,row,column,marker_type,px_per_hour_rate\n")
        for m in calibrated_marks:
            f.write(f"{m['date_utc']},{m['time_utc']},{m['date_local_utc3']},{m['time_local_utc3']},"
                    f"{m['row']},{m['column']},{m['marker_type']},{m['px_per_hour_rate']}\n")

    print(f"[ROBUST] Time marks calibrated with global closure ({global_rate:.2f} px/h): {csv_file}")
    return csv_file


def parse_magnetogram_date(file_path: str | Path) -> str:
    """
    Extracts the observation date (YYYY-MM-DD) from filenames like 'n640912.tif'.
    
    Handles 2-digit years according to Tatuoca's analog operational era (1959–2007):
      - 59 to 99 -> 1959 to 1999
      - 00 to 10 -> 2000 to 2010
    """
    path_obj = Path(file_path)
    # Search for exactly 6 consecutive digits in the filename stem
    match = re.search(r'(\d{6})', path_obj.stem)
    if not match:
        raise ValueError(f"Could not extract a 6-digit date pattern from '{path_obj.name}'.")

    date_str = match.group(1)
    yy = int(date_str[0:2])
    mm = int(date_str[2:4])
    dd = int(date_str[4:6])

    # Tatuoca opened in 1959; analog paper runs transition before 2008
    year = (1900 + yy) if yy >= 50 else (2000 + yy)

    # Validates calendar bounds (leap years, month ranges)
    parsed_date = datetime(year, mm, dd)
    return parsed_date.strftime("%Y-%m-%d")


def extract_start_label_roi(image, baseline_coords):
    """
    Crops the small text area immediately to the left and above the baseline start.
    """
    sorted_baseline = baseline_coords[np.argsort(baseline_coords[:, 1])]
    first_row, first_col = int(sorted_baseline[0, 0]), int(sorted_baseline[0, 1])

    # The label typically sits 20-80 px above the line and 0-80 px to the left
    r_min = max(0, first_row - 80)
    r_max = max(0, first_row - 10)
    c_min = max(0, first_col - 90)
    c_max = min(image.shape[1], first_col + 40)

    roi = image[r_min:r_max, c_min:c_max]
    return roi






def detect_start_period_cv(image: np.ndarray, baseline_coords: np.ndarray, output_dir: Path | None = None) -> str:
    """
    Detects if the record starts in the morning ('12 UT') or night ('00 UT'/'21')
    using OpenCV connected components and topological features.
    
    Returns:
        'morning' (12 UT) or 'night' (00 UT / 24 UT / 21h)
    """
    # 1. Crop ROI above the first baseline pixel
    sorted_baseline = baseline_coords[np.argsort(baseline_coords[:, 1])]
    first_row, first_col = int(sorted_baseline[0, 0]), int(sorted_baseline[0, 1])

    # Crop text ROI above the baseline start
    r_min = max(0, first_row - 85)
    r_max = max(0, first_row - 10)
    c_min = max(0, first_col - 90)
    c_max = min(image.shape[1], first_col + 30)

    roi = image[r_min:r_max, c_min:c_max]
    if roi.size == 0:
        return "morning"

    # Prepare debug BGR canvas
    debug_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # Threshold ink
    _, binary = cv2.threshold(roi, 160, 255, cv2.THRESH_BINARY_INV)

    # Detect character components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    chars = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if 40 < area < 2000 and h > 15:
            chars.append((x, y, w, h, area, i))

    if not chars:
        if output_dir:
            cv2.imwrite(str(output_dir / "debug_start_label.png"), debug_bgr)
        return "morning"

    # Sort left to right
    chars.sort(key=lambda c: c[0])

    # Draw bounding boxes and aspect ratios on all identified characters
    for idx, (x, y, w, h, area, label_idx) in enumerate(chars):
        ar = h / max(1, w)
        # Green bounding box
        cv2.rectangle(debug_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Red aspect ratio label above the box
        cv2.putText(
            debug_bgr,
            f"{ar:.1f}",
            (x, max(10, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1
        )

    # Analyze first character
    x, y, w, h, area, label_idx = chars[0]
    aspect_ratio = h / max(1, w)

    # Check for internal holes (e.g., inside '0')
    char_mask = (labels[y:y+h, x:x+w] == label_idx).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(char_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    has_hole = False
    if hierarchy is not None:
        for h_entry in hierarchy[0]:
            if h_entry[2] != -1:
                has_hole = True
                break

    # Decision rule
    if aspect_ratio >= 2.0 and not has_hole:
        decision = "morning"
    elif has_hole or aspect_ratio < 1.6:
        decision = "night"
    else:
        decision = "morning"

    # Add classification text to top-left of the debug image
    cv2.putText(
        debug_bgr,
        f"{decision.upper()}",
        (3, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 0, 0),
        1
    )

    # Save to the image's specific output folder
    if output_dir is not None:
        debug_path = output_dir / "debug_start_label.png"
        cv2.imwrite(str(debug_path), debug_bgr)

    return decision



def export_start_end_verification_banner(
    image: np.ndarray,
    baseline_coords: np.ndarray,
    output_dir: Path
) -> Path:
    """
    Creates a single side-by-side verification banner containing:
      - Left panel: Starting label (~12 UT / 00 UT) with green character boxes
      - Right panel: Ending label (~12 / 9) and baseline termination
    
    Saves:
        {output_dir}/start_and_end_verification.png
    """
    sorted_baseline = baseline_coords[np.argsort(baseline_coords[:, 1])]
    first_row, first_col = int(sorted_baseline[0, 0]), int(sorted_baseline[0, 1])
    last_row, last_col = int(sorted_baseline[-1, 0]), int(sorted_baseline[-1, 1])

    # 1. Start ROI
    r_min1 = max(0, first_row - 85)
    r_max1 = max(0, first_row - 10)
    c_min1 = max(0, first_col - 90)
    c_max1 = min(image.shape[1], first_col + 30)
    roi_start = image[r_min1:r_max1, c_min1:c_max1]
    bgr_start = cv2.cvtColor(roi_start, cv2.COLOR_GRAY2BGR)

    _, bin_start = cv2.threshold(roi_start, 160, 255, cv2.THRESH_BINARY_INV)
    n_start, _, stats_start, _ = cv2.connectedComponentsWithStats(bin_start)
    for i in range(1, n_start):
        x, y, w, h, area = stats_start[i]
        if 40 < area < 2000 and h > 15:
            cv2.rectangle(bgr_start, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # 2. End ROI
    r_min2 = max(0, last_row - 85)
    r_max2 = min(image.shape[0], last_row + 35)
    c_min2 = max(0, last_col - 100)
    c_max2 = min(image.shape[1], last_col + 60)
    roi_end = image[r_min2:r_max2, c_min2:c_max2]
    bgr_end = cv2.cvtColor(roi_end, cv2.COLOR_GRAY2BGR)

    _, bin_end = cv2.threshold(roi_end, 160, 255, cv2.THRESH_BINARY_INV)
    n_end, _, stats_end, _ = cv2.connectedComponentsWithStats(bin_end)
    for i in range(1, n_end):
        x, y, w, h, area = stats_end[i]
        if 25 < area < 2500 and h > 10 and w < 80:
            cv2.rectangle(bgr_end, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # 3. Match panel heights
    target_h = 140
    def scale_crop(crop, target_h):
        h, w = crop.shape[:2]
        new_w = int(w * (target_h / max(1, h)))
        return cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    s_img = scale_crop(bgr_start, target_h)
    e_img = scale_crop(bgr_end, target_h)

    # 4. Assemble canvas with header banners and vertical divider
    header_h = 40
    gutter = 12
    total_w = s_img.shape[1] + e_img.shape[1] + gutter
    canvas = np.full((target_h + header_h, total_w, 3), 245, dtype=np.uint8)

    # Vertical divider
    divider_x = s_img.shape[1] + (gutter // 2)
    canvas[:, divider_x] = (180, 180, 180)

    # Paste panels
    canvas[header_h:, :s_img.shape[1]] = s_img
    canvas[header_h:, s_img.shape[1] + gutter:] = e_img

    # Titles
    cv2.putText(canvas, "Start: 12 UT", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 160), 2)
    cv2.putText(canvas, "End: 12 UT (9h loc)", (s_img.shape[1] + gutter + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 160), 2)

    out_file = output_dir / "start_and_end_verification.png"
    cv2.imwrite(str(out_file), canvas)
    print(f"Combined verification banner saved to: {out_file}")
    return out_file




def overlay_approach_time_marks(
    base_image: np.ndarray,
    time_marks_csv: Path,
    output_path: Path,
    line_color: tuple = (0, 0, 255),  # BGR: Red for App 2, Blue (255, 0, 0) for App 3
    line_thickness: int = 2,
    annotate_hours: bool = True
) -> np.ndarray:
    """
    Overlays vertical lines for hourly marks from a generated time marks CSV
    onto the magnetogram image, printing hour labels at the top.
    """
    if not time_marks_csv.exists():
        return base_image

    df = pd.read_csv(time_marks_csv)

    # Convert to BGR if grayscale
    if len(base_image.shape) == 2:
        vis = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        vis = base_image.copy()

    h, w = vis.shape[:2]

    for _, row in df.iterrows():
        col = int(round(row['column']))
        if 0 <= col < w:
            mtype = row.get('marker_type', '')
            
            # First and last pixels in orange (BGR: 0, 165, 255)
            if mtype in ['first_pixel', 'last_pixel']:
                cv2.line(vis, (col, 0), (col, h), (0, 165, 255), 2, cv2.LINE_AA)
            else:
                cv2.line(vis, (col, 0), (col, h), line_color, line_thickness, cv2.LINE_AA)

            # Hour label at the top margin
            if annotate_hours and 'time_utc' in row and mtype == 'hourly_mark':
                hour_str = str(row['time_utc'])[:2] + "h"
                cv2.putText(
                    vis,
                    hour_str,
                    (max(0, col - 14), 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    line_color,
                    2,
                    cv2.LINE_AA
                )

    cv2.imwrite(str(output_path), vis)
    print(f"Overlay salvo em: {output_path}")
    return vis


def overlay_approach_comparison(
    base_image: np.ndarray,
    csv_app2: Path,
    csv_app3: Path,
    output_path: Path
) -> np.ndarray:
    """
    Overlays Approach 2 (Red) and Approach 3 (Blue) on the SAME image
    to highlight the physical drift across the 24-hour recording period.
    """
    if not csv_app2.exists() or not csv_app3.exists():
        return base_image

    df2 = pd.read_csv(csv_app2)
    df3 = pd.read_csv(csv_app3)

    if len(base_image.shape) == 2:
        vis = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        vis = base_image.copy()

    h, w = vis.shape[:2]

    # Approach 2 lines (Red)
    for _, r in df2.iterrows():
        col = int(round(r['column']))
        if 0 <= col < w and r.get('marker_type') == 'hourly_mark':
            cv2.line(vis, (col, 0), (col, h), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, str(r['time_utc'])[:2], (max(0, col - 10), 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # Approach 3 lines (Blue)
    for _, r in df3.iterrows():
        col = int(round(r['column']))
        if 0 <= col < w and r.get('marker_type') == 'hourly_mark':
            cv2.line(vis, (col, 0), (col, h), (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, str(r['time_utc'])[:2], (max(0, col - 10), 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    # Legend in the top-left margin
    cv2.putText(vis, "Approach 2 (Red): local early rate", (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "Approach 3 (Blue): long-baseline span", (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis)
    print(f"Comparativo visual salvo em: {output_path}")
    return vis


def export_hourly_trace_matches(
    time_marks_csv: Path,
    curve_coords: np.ndarray,
    baseline_coords: np.ndarray,
    output_csv: Path,
    px_to_mm: float = 0.1,
    component_name: str = "H"
):
    """
    Matches each calibrated 1h time mark to the exact trace coordinate at that column,
    exporting the final handoff table for the geophysics analysis workflow.
    """
    df_marks = pd.read_csv(time_marks_csv)

    # Convert curve and baseline arrays to lookup dictionaries: col -> row
    # In case of duplicate columns in the skeleton, take the median row
    df_curve = pd.DataFrame(curve_coords, columns=["row_trace", "col"])
    curve_lookup = df_curve.groupby("col")["row_trace"].median().to_dict()

    df_base = pd.DataFrame(baseline_coords, columns=["row_base", "col"])
    base_lookup = df_base.groupby("col")["row_base"].median().to_dict()
    mean_baseline_global = float(df_base["row_base"].mean())

    records = []
    for _, row in df_marks.iterrows():
        c = int(round(row["column"]))

        # Retrieve row positions; fallback to nearest column if skeleton has a 1px gap
        if c in curve_lookup:
            r_trace = curve_lookup[c]
        else:
            nearest_col = min(curve_lookup.keys(), key=lambda x: abs(x - c))
            r_trace = curve_lookup[nearest_col]

        # Use local baseline row if available, otherwise mean baseline
        r_base = base_lookup.get(c, mean_baseline_global)

        # Physical vertical distance in millimeters
        delta_mm = (r_base - r_trace) * px_to_mm

        records.append({
            "component": component_name,
            "date_utc": row["date_utc"],
            "time_utc": row["time_utc"],
            "date_local_utc3": row["date_local_utc3"],
            "time_local_utc3": row["time_local_utc3"],
            "column": c,
            "row_baseline": round(r_base, 1),
            "row_trace": round(r_trace, 1),
            "delta_height_mm": round(delta_mm, 2),
            "marker_type": row["marker_type"]
        })

    df_out = pd.DataFrame(records)
    df_out["datetime_utc"] = pd.to_datetime(df_out["date_utc"] + " " + df_out["time_utc"])
    df_out["datetime_local_utc3"] = pd.to_datetime(df_out["date_local_utc3"] + " " + df_out["time_local_utc3"])
    df_out.to_csv(output_csv, index=False)
    print(f"Tabela de alturas horárias exportada com sucesso: {output_csv}")







#### Define Main function
def main():
  """
  Função principal para processar as imagens.
  """
  shape, px_to_mm = magnetogram_image_scale(10)
  dataset = "gray_300"
  images_dir = Path(f"Images/{dataset}")
  
  #images_dir = Path("Images/gray_300")

  for output_dir, opened in ImageOpener(images_dir):
      print(f"Processando imagem: {output_dir.name}")

      # # Automatically parse the start date from the filename (e.g., '1964-09-12')
      # current_date = parse_magnetogram_date(output_dir.name)
      # print(f"Data detectada: {current_date}")

      # # Automatically detect whether sheet started in morning (12 UT) or night (00 UT)
      # detected_period = detect_start_period_cv(opened, base0)
      # print(f"Período detectado: {detected_period.upper()} ({'12 UT' if detected_period == 'morning' else '00 UT'})")

      cropped_image = cut_image(opened, shape, output_dir=output_dir)
      imwrite(str(output_dir / "cutted.png"), cropped_image)

      gauss_image = gauss_process(cropped_image)
      imwrite(str(output_dir / "gauss.png"), gauss_image)

      canny_image = canny_filter(gauss_image)
      white_canny_image = 255 - canny_image
      imwrite(str(output_dir / "canny.png"), canny_image)
      imwrite(str(output_dir / "white_canny.png"), white_canny_image)

      skeleton_image = skeletonize(canny_image, 15)
      white_skeleton_image = 255 - skeleton_image
      imwrite(str(output_dir / "skeleton.png"), skeleton_image)
      imwrite(str(output_dir / "white_skeleton.png"), white_skeleton_image)

      separated_clusters = dbscan_separation(skeleton_image)
      curves, baselines = line_separation(separated_clusters)

      save_lines(curves, output_dir, "curve", shape)
      save_lines(baselines, output_dir, "basel", shape)
      calculate_differences(curves, baselines, output_dir, px_to_mm)

      # ---------------------------------------------------------
      # TIME-MARK COMPUTATION & DOWNSTREAM HANDOFF EXPORT
      # ---------------------------------------------------------
      # if len(baselines) > 0: # versao original
      if len(baselines) > 0 and len(curves) > 0:

            # Currently working for H component
            base0 = baselines[0]  # baseline for H
            curve0 = curves[0]  # Trace curve of component H
            
            # Automatically parse the start date from the filename (e.g., '1964-09-12')
            current_date = parse_magnetogram_date(output_dir.name)

            # Automatically detect whether sheet started in morning (12 UT) or night (00 UT)
            # detected_period = detect_start_period_cv(
            #     image=opened,
            #     baseline_coords=base0,
            #     output_dir=output_dir
            # )

            # Inside main() loop:
            if "9101" in output_dir.name:
                detected_period = "night"
            else:
                detected_period = detect_start_period_cv(
                image=opened,
                baseline_coords=base0,
                output_dir=output_dir
            )

            print(f"Período detectado: {detected_period.upper()}")

            # Robust time-mark calibration with closure
            csv_app3 = compute_time_marks_robust(
                baseline_coords=base0,
                output_dir=output_dir,
                start_date=current_date,
                start_period=detected_period,
                search_window_px=14
            )

            # Visual overlay on cropped ROI
            overlay_approach_time_marks(
                base_image=cropped_image,
                time_marks_csv=csv_app3,
                output_path=output_dir / "approach3_timemarks_cutted.png",
                line_color=(255, 0, 0)  # Blue
            )

            overlay_approach_time_marks(
                base_image=canny_image,
                time_marks_csv=csv_app3,
                output_path=output_dir / "approach3_timemarks_canny.png",
                line_color=(255, 0, 0)  # Blue
            )

            overlay_approach_time_marks(
                            base_image=white_canny_image,
                            time_marks_csv=csv_app3,
                            output_path=output_dir / "approach3_timemarks_white_canny.png",
                            line_color=(255, 0, 0)  # Blue
                        )

            # Export hourly height deviations
            export_hourly_trace_matches(
                time_marks_csv=csv_app3,
                curve_coords=curve0,
                baseline_coords=base0,
                output_csv=output_dir / "hourly_H_deviations_approach3.csv",
                px_to_mm=px_to_mm,
                component_name="H"
            )

            #csv_app3 = output_dir / "lines" / "approach3_basel0_time_marks.csv"


      _30min = get_30min_values(canny_image, curves, baselines, output_dir, px_to_mm)
      imwrite(str(output_dir / "30min.png"), _30min)

      dbscan1 = overlay_lines(skeleton_image, curves)
      white_dbscan1 = overlay_lines(white_skeleton_image, curves)
      dbscan2 = overlay_lines(canny_image, curves)
      white_dbscan2 = overlay_lines(white_canny_image, curves)

      imwrite(str(output_dir / "dbscan1.png"), dbscan1)
      imwrite(str(output_dir / "white_dbscan1.png"), white_dbscan1)
      imwrite(str(output_dir / "dbscan2.png"), dbscan2)
      imwrite(str(output_dir / "white_dbscan2.png"), white_dbscan2)

      print(f"Concluído: {output_dir.name}\n----------------------\n")

  print("Processo Terminado!")

if __name__ == "__main__": 
   main()