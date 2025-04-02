import streamlit as st
import chess
import chess.svg
import os
import pickle
import torch
import numpy as np
from PIL import Image
import io
import base64
import sys
import os

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from the chess module
# Since the main module is named "chess copy.py" with a space, we need to use importlib
import importlib.util

spec = importlib.util.spec_from_file_location(
    "chessgame",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess copy.py"),
)
chessgame = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chessgame)

# Set page config
st.set_page_config(
    page_title="Chess AI Trainer",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Function to render chess board as an SVG and then convert to displayable image
def render_board(board, last_move=None):
    svg_board = chess.svg.board(
        board,
        size=400,
        lastmove=last_move,
        colors={"square light": "#FFCE9E", "square dark": "#D18B47"},
    )
    return svg_board


# Function to convert SVG to HTML for display
def svg_to_html(svg_str):
    b64 = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
    html = f'<img src="data:image/svg+xml;base64,{b64}" width="400" height="400"/>'
    return html


# Function to get valid moves for a square
def get_valid_moves(board, square):
    valid_moves = []
    for move in board.legal_moves:
        if move.from_square == square:
            valid_moves.append(move)
    return valid_moves


# Initialize session state
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "selected_square" not in st.session_state:
    st.session_state.selected_square = None
if "last_move" not in st.session_state:
    st.session_state.last_move = None
if "move_history" not in st.session_state:
    st.session_state.move_history = []
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "player_color" not in st.session_state:
    st.session_state.player_color = chess.WHITE


# Load the trained model
@st.cache_resource
def load_model():
    return chessgame.load_trained_model()


model = load_model()

# Main title
st.title("♟️ Chess AI Trainer")
st.markdown("Play against a custom-trained AI model")

# Sidebar with options
with st.sidebar:
    st.header("Game Options")

    # New game button
    if st.button("New Game"):
        st.session_state.board = chess.Board()
        st.session_state.selected_square = None
        st.session_state.last_move = None
        st.session_state.move_history = []
        st.session_state.game_over = False

    # Color selection for new games
    color_choice = st.radio("Play as:", ("White", "Black"))
    st.session_state.player_color = (
        chess.WHITE if color_choice == "White" else chess.BLACK
    )

    # AI difficulty/model selection
    ai_type = st.radio("AI Opponent:", ("GA Trained Model", "Stockfish"))

    # Move history
    st.header("Move History")
    move_history_text = "\n".join(
        [f"{i+1}. {move}" for i, move in enumerate(st.session_state.move_history)]
    )
    st.text_area("Moves:", value=move_history_text, height=300, disabled=True)

# Main board display
col1, col2 = st.columns([2, 1])

with col1:
    # Display board
    board_svg = render_board(st.session_state.board, st.session_state.last_move)
    st.components.v1.html(svg_to_html(board_svg), height=400, width=400)

    # Show game status
    if st.session_state.game_over:
        result = st.session_state.board.result()
        st.header(f"Game Over: {result}")
        if st.session_state.board.is_checkmate():
            st.subheader("Checkmate!")
        elif st.session_state.board.is_stalemate():
            st.subheader("Stalemate!")
        elif st.session_state.board.is_insufficient_material():
            st.subheader("Insufficient material!")
        elif st.session_state.board.is_fifty_moves():
            st.subheader("Fifty-move rule!")
        elif st.session_state.board.is_repetition():
            st.subheader("Threefold repetition!")
    else:
        st.header(
            f"{'White' if st.session_state.board.turn == chess.WHITE else 'Black'} to move"
        )

# Game interaction
with col2:
    st.header("Make a Move")

    # Create grid of buttons for selecting squares
    files = "abcdefgh"
    ranks = "87654321"

    # Display helper text
    if st.session_state.selected_square is None:
        st.text("Select a piece to move")
    else:
        from_square_name = chess.square_name(st.session_state.selected_square)
        st.text(f"Selected: {from_square_name}. Now select destination.")
        st.button(
            "Cancel Selection",
            on_click=lambda: setattr(st.session_state, "selected_square", None),
        )

    # Create a text input for moves in UCI format (alternative to clickable board)
    move_uci = st.text_input("Enter move in UCI format (e.g., e2e4):", "")

    if st.button("Make Move"):
        if move_uci:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in st.session_state.board.legal_moves:
                    st.session_state.board.push(move)
                    st.session_state.last_move = move
                    st.session_state.move_history.append(move_uci)
                    st.session_state.selected_square = None

                    # Check if game is over after player's move
                    if st.session_state.board.is_game_over():
                        st.session_state.game_over = True
                    else:
                        # AI's turn
                        if ai_type == "Stockfish":
                            try:
                                # The Stockfish path may need to be adjusted
                                stockfish_path = r"C:\Users\Krishna Bansal\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
                                if os.path.exists(stockfish_path):
                                    engine = chess.engine.SimpleEngine.popen_uci(
                                        stockfish_path
                                    )
                                    result = engine.play(
                                        st.session_state.board,
                                        chess.engine.Limit(time=0.1),
                                    )
                                    if result.move:
                                        st.session_state.board.push(result.move)
                                        st.session_state.last_move = result.move
                                        st.session_state.move_history.append(
                                            result.move.uci()
                                        )
                                    engine.quit()
                                else:
                                    st.error(
                                        "Stockfish engine not found. Please check the path."
                                    )
                            except Exception as e:
                                st.error(f"Error with Stockfish: {e}")
                        else:
                            # Use GA trained model
                            ai_move = chessgame.compute_personal_trainer_move(
                                st.session_state.board
                            )
                            if ai_move:
                                st.session_state.board.push(ai_move)
                                st.session_state.last_move = ai_move
                                st.session_state.move_history.append(ai_move.uci())

                        # Check if game is over after AI's move
                        if st.session_state.board.is_game_over():
                            st.session_state.game_over = True
                else:
                    st.error("Illegal move!")
            except ValueError:
                st.error("Invalid move format. Use UCI format (e.g., e2e4)")
        else:
            st.error("Please enter a move")

    # Show legal moves for the current player
    st.header("Legal Moves")
    legal_moves = [move.uci() for move in st.session_state.board.legal_moves]
    st.write(", ".join(legal_moves) if legal_moves else "No legal moves")

    # Start AI move button (for manual triggering)
    if st.button("Let AI Move"):
        if not st.session_state.board.is_game_over():
            if ai_type == "Stockfish":
                try:
                    stockfish_path = r"C:\Users\Krishna Bansal\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
                    if os.path.exists(stockfish_path):
                        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                        result = engine.play(
                            st.session_state.board, chess.engine.Limit(time=0.1)
                        )
                        if result.move:
                            st.session_state.board.push(result.move)
                            st.session_state.last_move = result.move
                            st.session_state.move_history.append(result.move.uci())
                        engine.quit()
                    else:
                        st.error("Stockfish engine not found. Please check the path.")
                except Exception as e:
                    st.error(f"Error with Stockfish: {e}")
            else:
                # Use GA trained model
                ai_move = chessgame.compute_personal_trainer_move(
                    st.session_state.board
                )
                if ai_move:
                    st.session_state.board.push(ai_move)
                    st.session_state.last_move = ai_move
                    st.session_state.move_history.append(ai_move.uci())

            # Check if game is over after AI's move
            if st.session_state.board.is_game_over():
                st.session_state.game_over = True

# Add information about the project
st.markdown("---")
st.markdown(
    """
## About this Chess AI Trainer
This application uses a neural network trained with a genetic algorithm to play chess. 
The AI has learned chess strategy through reinforcement learning and can adapt to different play styles.

### Features:
- Play against a custom-trained neural network
- Option to play against Stockfish engine
- Visual chessboard interface
- Move history tracking

Make your moves by entering them in UCI format (e.g., e2e4) in the text field.
"""
)

# Add footer
st.markdown("---")
st.markdown("Created with Streamlit ♟️")
