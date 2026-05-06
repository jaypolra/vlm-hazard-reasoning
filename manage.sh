#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Project Manager — SDI Hazard Recognition + ARIA VLM Demo
# ═══════════════════════════════════════════════════════════════

# ── Paths ───────────────────────────────────────────────────────
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOLO_BACKEND="$BASE/ai-hazard-recognition-backup-jay/backend"
YOLO_FRONTEND="$BASE/ai-hazard-recognition-backup-jay/frontend"
ARIA_DIR="$BASE/vlm_demo"

# ── Logs ────────────────────────────────────────────────────────────────────
# Use user-specific temp directory to avoid permission conflicts
LOG_DIR="${XDG_RUNTIME_DIR:-/tmp/user_$$}"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="/tmp"

# ── Conda ────────────────────────────────────────────────────────────────────
# Auto-detect conda executable (works on any machine regardless of install path)
if command -v conda &>/dev/null; then
    CONDA_EXE="$(command -v conda)"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/anaconda3/bin/conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_EXE="$HOME/miniconda3/bin/conda"
else
    echo "WARNING: conda not found. Service start functions will fail." >&2
    CONDA_EXE="conda"
fi

LOG_GEMMA="$LOG_DIR/gemma_$(whoami).log"
LOG_ARIA="$LOG_DIR/aria_$(whoami).log"
LOG_YOLO_BE="$LOG_DIR/yolo_backend_$(whoami).log"
LOG_YOLO_FE="$LOG_DIR/yolo_frontend_$(whoami).log"

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════

is_port_open() { curl -s -o /dev/null --connect-timeout 2 http://localhost:$1; }

status_dot() {
    if is_port_open $1; then
        echo -e "${GREEN}●${RESET} $2 (port $1)"
    else
        echo -e "${RED}○${RESET} $2 (port $1) — not running"
    fi
}

wait_for_port() {
    local port=$1 name=$2 timeout=${3:-60}
    echo -ne "  Waiting for $name"
    for i in $(seq 1 $timeout); do
        if is_port_open $port 2>/dev/null; then
            echo -e " ${GREEN}✓${RESET}"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    echo -e " ${RED}timeout${RESET}"
    return 1
}

stop_port() {
    local port=$1 name=$2
    local pids=$(fuser ${port}/tcp 2>/dev/null)
    if [ -n "$pids" ]; then
        fuser -k ${port}/tcp 2>/dev/null
        echo -e "  ${RED}stopped${RESET} $name (port $port)"
    else
        echo -e "  ${YELLOW}already stopped${RESET} $name (port $port)"
    fi
}

# ════════════════════════════════════════════════════════════════
#  Start functions
# ════════════════════════════════════════════════════════════════

start_gemma() {
    if is_port_open 8000; then
        echo -e "  ${YELLOW}already running${RESET} Gemma 3 27B (port 8000)"
        return
    fi
    echo -e "  ${CYAN}launching${RESET} Gemma 3 27B on GPU 0 → port 8000"
    nohup "$CONDA_EXE" run -n gemmaenv bash -c \
        "CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-3-27b-it \
        --port 8000 \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --enable-lora=false \
        --disable-log-stats" \
        > $LOG_GEMMA 2>&1 &
}

start_aria() {
    if is_port_open 8501; then
        echo -e "  ${YELLOW}already running${RESET} ARIA Streamlit (port 8501)"
        return
    fi
    echo -e "  ${CYAN}launching${RESET} ARIA demo → port 8501"
    nohup "$CONDA_EXE" run -n gemmaenv bash -c \
        "cd $ARIA_DIR && streamlit run app.py --server.port 8501 --server.address 0.0.0.0" \
        > $LOG_ARIA 2>&1 &
}

start_yolo_backend() {
    if is_port_open 8001; then
        echo -e "  ${YELLOW}already running${RESET} YOLO backend (port 8001)"
        return
    fi
    echo -e "  ${CYAN}launching${RESET} YOLO backend → port 8001"
    nohup "$CONDA_EXE" run -n hazardenv bash -c \
        "cd $YOLO_BACKEND && python -m uvicorn main:app --host 0.0.0.0 --port 8001" \
        > $LOG_YOLO_BE 2>&1 &
}

start_yolo_frontend() {
    if is_port_open 8502; then
        echo -e "  ${YELLOW}already running${RESET} YOLO frontend (port 8502)"
        return
    fi
    # Ensure cache dir is writable by all users (avoids EACCES for nexislab / others)
    mkdir -p "$YOLO_FRONTEND/node_modules/.cache"
    chmod -R 777 "$YOLO_FRONTEND/node_modules/.cache" 2>/dev/null
    echo -e "  ${CYAN}launching${RESET} YOLO React frontend → port 8502"
    nohup "$CONDA_EXE" run -n hazardenv bash -c \
        "cd $YOLO_FRONTEND && PORT=8502 CI=true BROWSER=none npm start" \
        > $LOG_YOLO_FE 2>&1 &
}

# Parallel startup with consolidated waiting
start_all_parallel() {
    echo -e "  ${BOLD}Starting all services in parallel...${RESET}"
    start_gemma
    start_aria
    start_yolo_backend
    start_yolo_frontend
    
    echo ""
    echo -e "  ${BOLD}Waiting for services to be ready...${RESET}"
    local ports=(8000 8001 8501 8502)
    local names=("Gemma 3 27B" "YOLO backend" "ARIA Streamlit" "YOLO frontend")
    local timeouts=(120 30 30 60)
    
    for i in "${!ports[@]}"; do
        wait_for_port "${ports[$i]}" "${names[$i]}" "${timeouts[$i]}" &
    done
    wait
}

# ════════════════════════════════════════════════════════════════
#  Stop functions
# ════════════════════════════════════════════════════════════════

stop_gemma()         { stop_port 8000 "Gemma 3 27B"; }
stop_aria()          { stop_port 8501 "ARIA Streamlit"; }
stop_yolo_backend()  { stop_port 8001 "YOLO backend"; }
stop_yolo_frontend() { stop_port 8502 "YOLO frontend"; }

# ════════════════════════════════════════════════════════════════
#  Status
# ════════════════════════════════════════════════════════════════

show_status() {
    echo ""
    echo -e "${BOLD}  Service Status${RESET}"
    echo "  ─────────────────────────────────"
    status_dot 8000 "Gemma 3 27B  (vLLM)"
    status_dot 8501 "ARIA demo    (Streamlit)"
    status_dot 8001 "YOLO backend (FastAPI)"
    status_dot 8502 "YOLO frontend(React)"
    echo ""
    echo -e "${BOLD}  GPU Memory${RESET}"
    echo "  ─────────────────────────────────"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free \
        --format=csv,noheader | awk -F',' \
        '{printf "  GPU %s |%s | used: %s | free: %s\n", $1,$2,$3,$4}'
    echo ""
}

# ════════════════════════════════════════════════════════════════
#  Help
# ════════════════════════════════════════════════════════════════

show_menu() {
    clear
    echo -e "${BOLD}${CYAN}"
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║     SDI Hazard Recognition + ARIA VLM Demo       ║"
    echo "  ╚══════════════════════════════════════════════════╝${RESET}"
    echo ""
    show_status
    echo -e "${BOLD}  ── Start ─────────────────────────────────────────${RESET}"
    echo "  1)  Start everything"
    echo "  2)  Start YOLO app         (backend + frontend)"
    echo "  3)  Start YOLO backend     only"
    echo "  4)  Start YOLO frontend    only"
    echo "  5)  Start ARIA app         (Gemma + Streamlit)"
    echo "  6)  Start Gemma 3 27B      only"
    echo "  7)  Start ARIA frontend    only"
    echo ""
    echo -e "${BOLD}  ── Stop ──────────────────────────────────────────${RESET}"
    echo "  8)  Stop everything"
    echo "  9)  Stop YOLO app"
    echo "  10) Stop ARIA app          (Gemma + Streamlit)"
    echo ""
    echo -e "${BOLD}  ── Other ─────────────────────────────────────────${RESET}"
    echo "  11) Restart everything"
    echo "  12) Show status + GPU memory"
    echo "  13) View logs"
    echo ""
    echo "  q)  Quit"
    echo ""
    echo -ne "${BOLD}  Enter choice: ${RESET}"
}

show_logs_menu() {
    echo ""
    echo -e "${BOLD}  Which log?${RESET}"
    echo "  1) Gemma 3 27B"
    echo "  2) ARIA Streamlit"
    echo "  3) YOLO backend"
    echo "  4) YOLO frontend"
    echo ""
    echo -ne "${BOLD}  Enter choice: ${RESET}"
    read log_choice
    case $log_choice in
        1) echo -e "\n${CYAN}Gemma log — Ctrl+C to exit${RESET}\n"; tail -f $LOG_GEMMA ;;
        2) echo -e "\n${CYAN}ARIA log — Ctrl+C to exit${RESET}\n";  tail -f $LOG_ARIA ;;
        3) echo -e "\n${CYAN}YOLO backend log — Ctrl+C to exit${RESET}\n"; tail -f $LOG_YOLO_BE ;;
        4) echo -e "\n${CYAN}YOLO frontend log — Ctrl+C to exit${RESET}\n"; tail -f $LOG_YOLO_FE ;;
        *) echo "Invalid choice" ;;
    esac
}

# ════════════════════════════════════════════════════════════════
#  Command-line interface
# ════════════════════════════════════════════════════════════════

show_cli_help() {
    echo -e "${BOLD}Usage:${RESET} ./manage.sh [COMMAND]"
    echo ""
    echo -e "${BOLD}Commands:${RESET}"
    echo "  start          Start all applications"
    echo "  stop           Stop all applications"
    echo "  restart        Restart all applications"
    echo "  status         Show service status and GPU memory"
    echo "  logs           Interactive logs menu"
    echo ""
    echo -e "${BOLD}Detailed start options:${RESET}"
    echo "  start all      Start everything"
    echo "  start yolo     Start YOLO backend + frontend"
    echo "  start aria     Start Gemma + ARIA Streamlit"
    echo "  start backend  Start YOLO backend only"
    echo "  start frontend Start YOLO frontend only"
    echo "  start gemma    Start Gemma 3 27B only"
    echo ""
    echo -e "${BOLD}Detailed stop options:${RESET}"
    echo "  stop all       Stop everything"
    echo "  stop yolo      Stop YOLO backend + frontend"
    echo "  stop aria      Stop Gemma + ARIA Streamlit"
    echo "  stop backend   Stop YOLO backend only"
    echo "  stop frontend  Stop YOLO frontend only"
    echo "  stop gemma     Stop Gemma 3 27B only"
    echo ""
    echo -e "${BOLD}Examples:${RESET}"
    echo "  ./manage.sh start           # Start all apps"
    echo "  ./manage.sh stop            # Stop all apps"
    echo "  ./manage.sh restart         # Restart all apps"
    echo "  ./manage.sh status          # Show status"
    echo "  ./manage.sh                 # Interactive menu"
    echo ""
    echo -e "${BOLD}Log files:${RESET}"
    echo "  Logs are stored in: $LOG_DIR"
    echo "  Each user has their own logs (e.g., gemma_\$(whoami).log)"
    echo ""
}

# ════════════════════════════════════════════════════════════════
#  Main entry point (CLI or interactive)
# ════════════════════════════════════════════════════════════════

if [ $# -eq 0 ]; then
    # No arguments — use interactive menu
    while true; do
        show_menu
        read choice
        echo ""

        case $choice in
            1)
                echo -e "${BOLD}Starting everything...${RESET}"
                start_all_parallel
                ;;
            2)
                echo -e "${BOLD}Starting YOLO app...${RESET}"
                start_yolo_backend &
                start_yolo_frontend
                wait
                echo -e "  ${GREEN}✓ YOLO app ready${RESET}"
                ;;
            3)
                echo -e "${BOLD}Starting YOLO backend...${RESET}"
                start_yolo_backend
                wait_for_port 8001 "YOLO backend" 30
                ;;
            4)
                echo -e "${BOLD}Starting YOLO frontend...${RESET}"
                start_yolo_frontend
                wait_for_port 8502 "YOLO frontend" 60
                ;;
            5)
                echo -e "${BOLD}Starting ARIA app...${RESET}"
                start_gemma &
                start_aria
                wait_for_port 8000 "Gemma 3 27B" 120 &
                wait_for_port 8501 "ARIA Streamlit" 30
                wait
                ;;
            6)
                echo -e "${BOLD}Starting Gemma 3 27B...${RESET}"
                start_gemma
                wait_for_port 8000 "Gemma 3 27B" 120
                ;;
            7)
                echo -e "${BOLD}Starting ARIA frontend...${RESET}"
                start_aria
                wait_for_port 8501 "ARIA Streamlit" 30
                ;;
            8)
                echo -e "${BOLD}Stopping everything...${RESET}"
                stop_gemma; stop_aria; stop_yolo_backend; stop_yolo_frontend
                ;;
            9)
                echo -e "${BOLD}Stopping YOLO app...${RESET}"
                stop_yolo_backend; stop_yolo_frontend
                ;;
            10)
                echo -e "${BOLD}Stopping ARIA app...${RESET}"
                stop_gemma; stop_aria
                ;;
            11)
                echo -e "${BOLD}Restarting everything...${RESET}"
                stop_gemma; stop_aria; stop_yolo_backend; stop_yolo_frontend
                sleep 2
                start_all_parallel
                ;;
            12)
                show_status
                ;;
            13)
                show_logs_menu
                ;;
            q|Q)
                echo -e "${GREEN}Bye!${RESET}\n"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please try again.${RESET}"
                ;;
        esac

        echo ""
        echo -ne "${BOLD}Press Enter to return to menu...${RESET}"
        read
    done
    # Command-line mode
    case "$1" in
        start)
            case "${2:-all}" in
                all)
                    echo -e "${BOLD}Starting everything...${RESET}"
                    start_all_parallel
                    ;;
                yolo)
                    echo -e "${BOLD}Starting YOLO app...${RESET}"
                    start_yolo_backend &
                    start_yolo_frontend
                    wait
                    wait_for_port 8001 "YOLO backend" 30 &
                    wait_for_port 8502 "YOLO frontend" 60
                    wait
                    ;;
                aria)
                    echo -e "${BOLD}Starting ARIA app...${RESET}"
                    start_gemma &
                    start_aria
                    wait_for_port 8000 "Gemma 3 27B" 120 &
                    wait_for_port 8501 "ARIA Streamlit" 30
                    wait
                    ;;
                backend)
                    echo -e "${BOLD}Starting YOLO backend...${RESET}"
                    start_yolo_backend
                    wait_for_port 8001 "YOLO backend" 30
                    ;;
                frontend)
                    echo -e "${BOLD}Starting YOLO frontend...${RESET}"
                    start_yolo_frontend
                    wait_for_port 8502 "YOLO frontend" 60
                    ;;
                gemma)
                    echo -e "${BOLD}Starting Gemma 3 27B...${RESET}"
                    start_gemma
                    wait_for_port 8000 "Gemma 3 27B" 120
                    ;;
                *)
                    echo -e "${RED}Invalid start option: $2${RESET}"
                    show_cli_help
                    exit 1
                    ;;
            esac
            ;;
        stop)
            case "${2:-all}" in
                all)
                    echo -e "${BOLD}Stopping everything...${RESET}"
                    stop_gemma; stop_aria; stop_yolo_backend; stop_yolo_frontend
                    ;;
                yolo)
                    echo -e "${BOLD}Stopping YOLO app...${RESET}"
                    stop_yolo_backend; stop_yolo_frontend
                    ;;
                aria)
                    echo -e "${BOLD}Stopping ARIA app...${RESET}"
                    stop_gemma; stop_aria
                    ;;
                backend)
                    echo -e "${BOLD}Stopping YOLO backend...${RESET}"
                    stop_yolo_backend
                    ;;
                frontend)
                    echo -e "${BOLD}Stopping YOLO frontend...${RESET}"
                    stop_yolo_frontend
                    ;;
                gemma)
                    echo -e "${BOLD}Stopping Gemma 3 27B...${RESET}"
                    stop_gemma
                    ;;
                *)
                    echo -e "${RED}Invalid stop option: $2${RESET}"
                    show_cli_help
                    exit 1
                    ;;
            esac
            ;;
        restart)
            case "${2:-all}" in
                all)
                    echo -e "${BOLD}Restarting everything...${RESET}"
                    stop_gemma; stop_aria; stop_yolo_backend; stop_yolo_frontend
                    sleep 2
                    start_all_parallel
                    ;;
                yolo)
                    echo -e "${BOLD}Restarting YOLO app...${RESET}"
                    stop_yolo_backend; stop_yolo_frontend
                    sleep 1
                    start_yolo_backend &
                    start_yolo_frontend
                    wait
                    wait_for_port 8001 "YOLO backend" 30 &
                    wait_for_port 8502 "YOLO frontend" 60
                    wait
                    ;;
                aria)
                    echo -e "${BOLD}Restarting ARIA app...${RESET}"
                    stop_gemma; stop_aria
                    sleep 1
                    start_gemma &
                    start_aria
                    wait_for_port 8000 "Gemma 3 27B" 120 &
                    wait_for_port 8501 "ARIA Streamlit" 30
                    wait
                    ;;
                *)
                    echo -e "${RED}Invalid restart option: $2${RESET}"
                    show_cli_help
                    exit 1
                    ;;
            esac
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs_menu
            ;;
        help|-h|--help)
            show_cli_help
            ;;
        *)
            echo -e "${RED}Unknown command: $1${RESET}"
            show_cli_help
            exit 1
            ;;
    esac
fi
