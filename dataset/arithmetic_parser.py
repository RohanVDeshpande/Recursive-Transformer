""" Adapted from Georg Nebehay's arithmetic parser at https://github.com/gnebehay/parser
Extended to include:
- multi-digit numbers
- decimals/floating point numbers
- explicit negative numbers
- string output, including parentheses (which aren't strictly necessary in the AST)
- single step reduction
"""

import sys
import enum
import re
import operator
import random

class TokenType(enum.Enum):
  T_NUM = 0
  T_PLUS = 1
  T_MINUS = 2
  T_MULT = 3
  T_DIV = 4
  T_LPAR = 5
  T_RPAR = 6
  T_END = 7

operations = {
    TokenType.T_PLUS: operator.add,
    TokenType.T_MINUS: operator.sub,
    TokenType.T_MULT: operator.mul,
    TokenType.T_DIV: operator.truediv
}

class Node:
  def __init__(self, token_type: TokenType, value=None):
    self.token_type = token_type
    self.value = value
    self.children = []
    self.paren = False

  def to_string(self, debug=False) -> str:
    """ Converts the AST to a string representation. Enable debug for readability (spaces).
    """
    s = str(self.value)
    if self.token_type != TokenType.T_NUM:
      sep = ' ' if debug else ''
      left_str = self.children[0].to_string(debug)
      right_str = self.children[1].to_string(debug)
      s = left_str + sep + str(self.value) + sep + right_str

    if self.paren:
      s = '(' + s + ')'

    return s

  def is_reducable(self) -> bool:
    """ Convenience function to determine whether this Node is reducable.
    """
    return self.token_type != TokenType.T_NUM or self.paren

  def step(self):
    """ Reduces the expression represented by this Node by randomly selecting one of its children
    to reduce if they are reducable. Otherwise, if both children are fully reduced, computes the 
    numerical value of this Node.
    """
    if self.token_type == TokenType.T_NUM:
      self.paren = False
      return
    
    # If both children are numeric, reduce this node
    left_reducable = self.children[0].is_reducable()
    right_reducable = self.children[1].is_reducable()
    if not (left_reducable or right_reducable):
      assert(self.children[0].token_type == TokenType.T_NUM)
      assert(self.children[1].token_type == TokenType.T_NUM)

      left_val = self.children[0].value
      right_val = self.children[1].value
      operation = operations[self.token_type]

      self.token_type = TokenType.T_NUM
      self.value = operation(left_val, right_val)
      if int(self.value) == self.value:
        self.value = int(self.value)
      self.children = []
      return

    # Otherwise, pick one of the children to reduce
    child_ind = 0 if left_reducable else 1
    if left_reducable and right_reducable:
      child_ind = random.randint(0, 1)
    self.children[child_ind].step()


def lexical_analysis(s: str) -> [Node]:
  """ Converts an arithmetic string into a series of tokens, represented by AST nodes directly.
  """
  mappings = {
    '+': TokenType.T_PLUS,
    '-': TokenType.T_MINUS,
    '*': TokenType.T_MULT,
    '/': TokenType.T_DIV,
    '(': TokenType.T_LPAR,
    ')': TokenType.T_RPAR}

  tokens = []
  idx = 0
  is_prev_value = False
  while idx < len(s):
    c = s[idx]
    if (not is_prev_value and (c == '+' or c == '-')) or re.match(r'\d', c): # handle numeric
      start = idx
      if c == '+' or c == '-': # account for explicit positive/negative number
        idx += 1

      while idx < len(s) and re.fullmatch(r'(\+|-)?(\d+(\.\d*)?|\.\d+)', s[start:idx + 1]):
        idx += 1
      
      num = float(s[start:idx])
      if int(num) == float(num):
        num = int(num)
      token = Node(TokenType.T_NUM, value=num)
      is_prev_value = True

    elif c in mappings:
      token_type = mappings[c]
      token = Node(token_type, value=c)
      idx += 1
      is_prev_value = (c == ')')
    else:
      raise Exception('Invalid token: {}'.format(c))
    tokens.append(token)
  tokens.append(Node(TokenType.T_END))
  return tokens


def match(tokens: [Node], token: TokenType) -> Node:
  """ Pops the next token off the stack if it matches the given Token.
  Used primarily to match parse parentheses.
  """
  if tokens[0].token_type == token:
    return tokens.pop(0)
  else:
    raise Exception('Invalid syntax on token {}: {}, expected {}'.format(tokens[0].token_type, tokens[0].value, token))

def parse_e(tokens: [Node]) -> Node:
  """ Parses lowest priority expressions: addition, subtraction.
  """
  left_node = parse_e2(tokens)

  while tokens[0].token_type in [TokenType.T_PLUS, TokenType.T_MINUS]:
    node = tokens.pop(0)
    node.children.append(left_node)
    node.children.append(parse_e2(tokens))
    left_node = node

  return left_node


def parse_e2(tokens: [Node]) -> Node:
  """ Parses second-priority expressions: multiplication, division
  """
  left_node = parse_e3(tokens)

  while tokens[0].token_type in [TokenType.T_MULT, TokenType.T_DIV]:
    node = tokens.pop(0)
    node.children.append(left_node)
    node.children.append(parse_e3(tokens))
    left_node = node

  return left_node


def parse_e3(tokens: [Node]) -> Node:
  """ Parses highest-priority expressions: parentheses and numeric values.
  """
  if tokens[0].token_type == TokenType.T_NUM:
    return tokens.pop(0)

  match(tokens, TokenType.T_LPAR)
  expression = parse_e(tokens)
  match(tokens, TokenType.T_RPAR)

  expression.paren = True 
  return expression


def parse(inputstring: str) -> Node:
  """ Parses an arithmetic string into an abstract syntax tree.
  """
  tokens = lexical_analysis(inputstring)
  ast = parse_e(tokens)
  match(tokens, TokenType.T_END)
  return ast

def generate_datapoints(inputstring: str, debug=False) -> [(str, str, bool)]:
  """ Generates a list of (expr, next_expr, finished) tuples for a given arithmetic expression.
  """
  datapoints = []
  ast = parse(inputstring)

  while ast.is_reducable():
    curr = ast.to_string(debug)
    ast.step()
    reduced = ast.to_string(debug)
    finished = not ast.is_reducable()

    datapoints.append((curr, reduced, finished))
  
  return datapoints

if __name__ == '__main__':
  inp = ''.join(sys.argv[1].split())
  # ast = parse(inp)
  for datapoint in generate_datapoints(inp, debug=True):
    print(datapoint)