# Adapted from Georg Nebehay's arithmetic parser at https://github.com/gnebehay/parser
import sys
import enum
import re

class TokenType(enum.Enum):
  T_NUM = 0
  T_PLUS = 1
  T_MINUS = 2
  T_MULT = 3
  T_DIV = 4
  T_LPAR = 5
  T_RPAR = 6
  T_END = 7

class Node:
  def __init__(self, token_type, value=None):
    self.token_type = token_type
    self.value = value
    self.children = []
    self.paren = False

  def to_string(self) -> str:
    left_str = self.children[0].to_string() if len(self.children) > 0 else ''
    right_str = self.children[1].to_string() if len(self.children) > 1 else ''

    s = left_str + str(self.value) + right_str
    if self.paren:
      s = '(' + s + ')'
    
    return s
    


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
      while idx < len(s) and re.match(r'\d', s[idx]):
        idx += 1
      token = Node(TokenType.T_NUM, value=int(s[start:idx]))
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