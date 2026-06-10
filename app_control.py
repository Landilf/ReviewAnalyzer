from __future__ import annotations

import streamlit as st


def request_cancel() -> None:
    st.session_state["cancel_requested"] = True


def clear_cancel() -> None:
    st.session_state["cancel_requested"] = False


def is_cancel_requested() -> bool:
    return bool(st.session_state.get("cancel_requested", False))


def set_operation_running(value: bool) -> None:
    st.session_state["operation_running"] = value


def is_operation_running() -> bool:
    return bool(st.session_state.get("operation_running", False))
