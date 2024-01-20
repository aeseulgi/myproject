# R as a calculator
3 + 2 # addition
3 - 2 # subtraction
3 * 2 # multiplication
3 / 2 # division
3 ^ 2 # power
9 %% 2 # remainder
9 %/% 2 # quotient

1 / 200 * 30
(59 + 73 + 2) / 3
sin(pi / 2)

0 / 0
class(NaN)

# Variables
r_rocks <- 2 ^ 3   # This is preferred. RStudio shortcut: Alt + - (minus sign).
r_rocks = 2 ^ 3    # This is possible, but it will cause confusion later. Avoid using =.
2 ^ 3 -> r_rocks   # This works but is weird. Avoid using ->.

r_rocks  # Inspecting a variable.
r_rokcs  # Watch out typos.
R_rocks  # Names are case-sensitive.

# Calling functions
y <- seq(1, 10, length.out = 5)
y
(y <- seq(1, 10, length.out = 5))

my_add <- function(x, y) {
  return(x+y)
}
my_add(3,2)
