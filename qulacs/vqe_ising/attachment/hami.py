import sys
from openfermion.ops import QubitOperator
from openfermion.linalg import eigenspectrum

def create_qubit_operator(nqubit):
  hami = QubitOperator()
  for i in range(nqubit):
    hami = hami + QubitOperator('X' + str(i))
    for j in range(i+1, nqubit):
      if i+1 == j:
        hami = hami + QubitOperator('Z' + str(i) + ' ' + 'Z' + str(j))

  return hami

if __name__ == '__main__':
  args = sys.argv
  nqubit = args[1]
  hami = create_qubit_operator(int(nqubit))
  print(eigenspectrum(hami))
