# Adapted from Georg Nebehay's arithmetic parser at https://github.com/gnebehay/parser
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

  def to_string(self) -> str:
    s = str(self.value)
    if self.token_type != TokenType.T_NUM:
      left_str = self.children[0].to_string()
      right_str = self.children[1].to_string()
      s = left_str + ' ' + str(self.value) + ' ' + right_str

    if self.paren:
      s = '(' + s + ')'

    return s

  def is_reducable(self) -> bool:
    return self.token_type != TokenType.T_NUM or self.paren

  def step(self):
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
      self.children = []
      return

    # Otherwise, pick one of the children to reduce
    child_ind = 0 if left_reducable else 1
    if left_reducable and right_reducable:
      child_ind = random.randint(0, 1)
    self.children[child_ind].step()


def lexical_analysis(s: str) -> [Node]:
  mappings = {
    '+': TokenType.T_PLUS,
    '-': TokenType.T_MINUS,
    '*': TokenType.T_MULT,
    '/': TokenType.T_DIV,
    '(': TokenType.T_LPAR,
    ')': TokenType.T_RPAR}

  tokens = []
  idx = 0
  while idx < len(s):
    c = s[idx]
    if c in mappings:
      token_type = mappings[c]
      token = Node(token_type, value=c)
      idx += 1
    elif re.match(r'\d', c):
      start = idx

      while idx < len(s) and re.fullmatch(r'\d+(\.\d*)?|\.\d+', s[start:idx + 1]):
        idx += 1
      
      num = s[start:idx]
      value = int(num) if re.fullmatch(r'\d+', num) else float(num)
      token = Node(TokenType.T_NUM, value=value)
    else:
      raise Exception('Invalid token: {}'.format(c))
    tokens.append(token)
  tokens.append(Node(TokenType.T_END))
  return tokens


def match(tokens: [Node], token: TokenType) -> Node:
  if tokens[0].token_type == token:
    return tokens.pop(0)
  else:
    raise Exception('Invalid syntax on token {}: {}, expected {}'.format(tokens[0].token_type, tokens[0].value, token))

def parse_e(tokens: [Node]) -> Node:
  left_node = parse_e2(tokens)

  while tokens[0].token_type in [TokenType.T_PLUS, TokenType.T_MINUS]:
    node = tokens.pop(0)
    node.children.append(left_node)
    node.children.append(parse_e2(tokens))
    left_node = node

  return left_node


def parse_e2(tokens: [Node]) -> Node:
  left_node = parse_e3(tokens)

  while tokens[0].token_type in [TokenType.T_MULT, TokenType.T_DIV]:
    node = tokens.pop(0)
    node.children.append(left_node)
    node.children.append(parse_e3(tokens))
    left_node = node

  return left_node


def parse_e3(tokens: [Node]) -> Node:
  if tokens[0].token_type == TokenType.T_NUM:
    return tokens.pop(0)

  match(tokens, TokenType.T_LPAR)
  expression = parse_e(tokens)
  match(tokens, TokenType.T_RPAR)

  expression.paren = True 
  return expression


def parse(inputstring: str) -> Node:
  tokens = lexical_analysis(inputstring)
  ast = parse_e(tokens)
  match(tokens, TokenType.T_END)
  return ast

if __name__ == '__main__':
  inp = ''.join(sys.argv[1].split())
  ast = parse(inp)
  print(ast.to_string())
  while ast.token_type != TokenType.T_NUM or ast.paren:
    ast.step()
    print(ast.to_string())