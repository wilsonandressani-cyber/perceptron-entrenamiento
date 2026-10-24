
# Entrenamiento del perceptron simple
# Basado en el ejercicio del Ing. Edwin Mejia
# Fecha: 24 de octubre de 2026

#' Entrenar un perceptron simple
#'
#' Entrena un clasificador binario mediante
#' el algoritmo del perceptron.
#'
#' @param X Matriz o data.frame de variables predictoras.
#' @param y Vector de clases binarias (0 y 1).
#' @param epocas Numero maximo de epocas.
#' @param tasa Tasa de aprendizaje.
#'
#' @return Lista con pesos, sesgo, medias y desviaciones.
#' @export
#'
#' @examples
#' X <- data.frame(
#'   horas = c(2, 4, 6, 8),
#'   asistencia = c(50, 60, 80, 95)
#' )
#' y <- c(0, 0, 1, 1)
#' modelo <- entrenar_perceptron(X, y)

entrenar_perceptron <- function(X, y, epocas = 100, tasa = 0.01) {

  X <- as.matrix(X)
  y <- as.numeric(y)

  if (!is.numeric(X) || any(!is.finite(X))) {
    stop("X debe contener valores numericos finitos.")
  }

  if (nrow(X) < 2 || ncol(X) < 1) {
    stop("X debe tener al menos dos filas y una columna.")
  }

  if (length(y) != nrow(X) ||
      anyNA(y) || !all(y %in% c(0, 1))) {
    stop("y debe contener una clase 0 o 1 por fila.")
  }

  if (length(epocas) != 1 || !is.finite(epocas) ||
      epocas < 1 || epocas != floor(epocas)) {
    stop("epocas debe ser un entero positivo.")
  }

  if (length(tasa) != 1 || !is.finite(tasa) || tasa <= 0) {
    stop("tasa debe ser un numero positivo.")
  }

  # Estandarizar variables
  medias <- colMeans(X)
  desviaciones <- apply(X, 2, sd)
  desviaciones[desviaciones == 0] <- 1

  Xs <- scale(X, center = medias, scale = desviaciones)

  # Inicializar pesos y sesgo
  pesos <- rep(0, ncol(Xs))
  sesgo <- 0

  # Entrenar el modelo
  for (e in seq_len(epocas)) {
    for (i in sample(seq_len(nrow(Xs)))) {

      z <- sum(Xs[i, ] * pesos) + sesgo
      pred <- as.integer(z >= 0)
      error <- y[i] - pred

      pesos <- pesos + tasa * error * Xs[i, ]
      sesgo <- sesgo + tasa * error
    }
  }

  return(list(
    pesos = pesos,
    sesgo = sesgo,
    medias = medias,
    desviaciones = desviaciones
  ))
}
