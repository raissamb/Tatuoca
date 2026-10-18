from numpy import (ndarray, dtype, int64,
  float64, zeros, vstack, errstate, arange)
from os.path import isfile, isdir, join, splitext
from os import mkdir, scandir
from itertools import pairwise, product
from cv2.typing import MatLike, Size
from cv2 import (imread, imwrite, resize, Canny, cvtColor,
  GaussianBlur, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
  adaptiveThreshold, COLOR_BGR2GRAY, COLOR_GRAY2BGR)
from lpmalgos import Ellipsoid, find_clusters
from skeleton import skeletonize
from shutil import rmtree

NDArray = ndarray[tuple[int, int], dtype[int64 | float64]]
BLACK, WHITE, RED, GREEN = 0, 255, (0, 0, 255), (0, 255, 0)
NDArrays = list[NDArray]

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
    if not isdir(path) and not isfile(path):
      raise FileNotFoundError(f"{path!r} não existe!")
    self.path, self.index = path, None

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
    def open_image_core(path: str):
      name, extension = splitext(path)
      valid_extensions = (".bmp", ".dib", ".jpeg", ".jpg",
        ".jpe", ".jp2", ".png", ".pnm", ".avif", ".pbm",
        ".pgm", ".ppm", ".pxm", ".webp", ".pfm", ".sr",
        ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic")
      if extension not in valid_extensions:
        raise TypeError(f"{path!r} não é uma imagem!")
      #if image := imread(path) is None:
      #  raise TypeError(f"{path!r} não pôde ser aberto!")
      image = imread(path, 0)  # Read directly as single-channel grayscale
      if image is None:
        raise TypeError(f"{path!r} não pôde ser aberto!")
      
      if isdir(name): rmtree(name)
      return name, bgr2gray(image)

    path, index = self.path, self.index
    if isfile(path):
      yield open_image_core(path)
      return
    if isinstance(index, int):
      files = [e for e in scandir(path) if e.is_file()]
      yield open_image_core(join(path, files[index].name))
      return
    if isinstance(index, slice):
      yield from (open_image_core(join(path, e.name)) for e in
        [e for e in scandir(path) if e.is_file()][index])
      return
    if index is None:
      yield from (open_image_core(join(path, e.name))
        for e in scandir(path) if e.is_file())
      return

def cut_image(image: MatLike, shape: Size) -> MatLike:
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
  #x1, x2, y1, y2 = 180, 2445, 305, 6095 # original
  #cropped = image[x1:x2, y1:y2] # original

  # Typical slice format: cropped = image[y_min:y_max, x_min:x_max]
  # Example relative crop (or verify pixel constants used in your repo):
  h, w = image.shape[:2]
  cropped = image[int(h * 0.15):int(h * 0.92), int(w * 0.05):int(w * 0.95)]
  
  resized = resize(cropped, shape[::-1])
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

def save_lines(lines: NDArrays, directory: str, dtype: str, shape: Size):
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
  lines_dir = f"{directory}/lines"
  black_image = zeros(shape, "uint8")
  if not isdir(lines_dir): mkdir(lines_dir)
  for i, subarray in enumerate(lines):
    image, (dim1, dim2) = black_image.copy(), subarray.T
    image[dim1, dim2] = WHITE
    # imwrite(fr"{lines_dir}\{dtype}{i}.png", image)
    # with open(fr"{lines_dir}\{dtype}{i}.txt", "w") as output:
    
    # Change:
    # imwrite(fr"{lines_dir}\{dtype}{i}.png", image)
    # with open(fr"{lines_dir}\{dtype}{i}.txt", "w") as output:
    # To:
    imwrite(f"{lines_dir}/{dtype}{i}.png", image)
    with open(f"{lines_dir}/{dtype}{i}.txt", "w") as output:
      output.write(f"Tamanho da imagem: {shape}\n")
      output.writelines(f"{elem}\n" for elem in subarray)
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

def calculate_differences(curves: NDArrays, baselines: NDArrays, directory: str, px_to_mm: float):
  """
  Calcula diferenças entre curvas e bases.

  Args:
    curves (NDArrays): Lista de curvas.
    baselines (NDArrays): Lista de bases.
    directory (str): Diretório para salvar os arquivos.
    px_to_mm (float): Constante de conversão.
  """
  differences_dir = f"{directory}/diffs"
  if not isdir(differences_dir): mkdir(differences_dir)
  curves_and_baselines = product(enumerate(curves), enumerate(baselines))

  for (curve_index, curve), (baseline_index, baseline) in curves_and_baselines:
    mean = baseline.T[0].mean(None, int)
    values = ((mean - curve.T[0]) * px_to_mm).round(1)
    #with open(fr"{differences_dir}\curve{curve_index}base{baseline_index}.txt", "w") as output: # old line

    filepath = f"{differences_dir}/curve{curve_index}base{baseline_index}.csv"
    with open(filepath, "w") as output:
      # Descriptive header
      output.write("row_trace,row_baseline_mean,column,distance_mm\n")
      # Numerical CSV values
      for (e1, e2), value in zip(curve, values):
        output.write(f"{e1},{mean},{e2},{value:.1f}\n")

  print("calculate_differences terminado!")
    

    # This part is working
  #   # Change:
  #   # with open(fr"{differences_dir}\curve{curve_index}base{baseline_index}.txt", "w") as output:
  #   # To:
  #   with open(f"{differences_dir}/curve{curve_index}base{baseline_index}.txt", "w") as output:
  #     output.writelines(f"({e1}, {mean}), {e2} - {value}\n" for (e1, e2), value in zip(curve, values))
  # print("calculate_differences terminado!")



def get_30min_values(image: MatLike, curves: NDArrays,
                     baselines: NDArrays, directory: str, px_to_mm: float):
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
  dir_30min = f"{directory}/30min"
  if not isdir(dir_30min): mkdir(dir_30min)
  for i, (curve, baseline) in enumerate(product(curves, baselines)):
    ybase, xbase = baseline.T
    ymean = ybase.mean().round().astype(int64)
    xpass = (xbase[-1] - xbase[0]) / 48
    xvalues = ((arange(49) * xpass) + xbase[0]).round().astype(int64)
    indexes = (curve[:, 1, None] == xvalues).any(1)
    yvalues = ((ymean - curve[indexes, 0]) * px_to_mm).round(1)
    with open(fr"{dir_30min}/30min{i}.txt", "w") as outfile:
      outfile.writelines(f"{xvalue}, {yvalue}\n" for xvalue, yvalue in zip(xvalues, yvalues))
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

def main():
  """
  Função principal para processar as imagens.
  """
  shape, px_to_mm = magnetogram_image_scale(10)
  for name, opened in ImageOpener("Images"):
    print(f"Imagem atual: {name}")
    if not isdir(name): mkdir(name)

    cropped_image = cut_image(opened, shape)
    imwrite(f"{name}/cutted.png", cropped_image)
    gauss_image = gauss_process(cropped_image)
    imwrite(f"{name}/gauss.png", gauss_image)

    canny_image = canny_filter(gauss_image)
    white_canny_image = 255 - canny_image
    imwrite(f"{name}/canny.png", canny_image)
    imwrite(f"{name}/white_canny.png", white_canny_image)

    skeleton_image = skeletonize(canny_image, 15)
    white_skeleton_image = 255 - skeleton_image
    imwrite(f"{name}/skeleton.png", skeleton_image)
    imwrite(f"{name}/white_skeleton.png", white_skeleton_image)

    separated_clusters = dbscan_separation(skeleton_image)
    curves, baselines = line_separation(separated_clusters)

    save_lines(curves, name, "curve", shape)
    save_lines(baselines, name, "basel", shape)
    calculate_differences(curves, baselines, name, px_to_mm)

    _30min = get_30min_values(canny_image, curves, baselines, name, px_to_mm)
    imwrite(f"{name}/30min.png", _30min)

    dbscan1 = overlay_lines(skeleton_image, curves)
    white_dbscan1 = overlay_lines(white_skeleton_image, curves)
    dbscan2 = overlay_lines(canny_image, curves)
    white_dbscan2 = overlay_lines(white_canny_image, curves)
    imwrite(f"{name}/dbscan1.png", dbscan1)
    imwrite(f"{name}/white_dbscan1.png", white_dbscan1)
    imwrite(f"{name}/dbscan2.png", dbscan2)
    imwrite(f"{name}/white_dbscan2.png", white_dbscan2)
    print("\n----------------------\n")

  print("Processo Terminado!")

if __name__ == "__main__": main()