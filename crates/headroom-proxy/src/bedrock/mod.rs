//! Native AWS Bedrock InvokeModel route — Phase D PR-D1.
//!
//! # Why a separate module?
//!
//! The Python proxy currently routes Anthropic-on-Bedrock through the
//! `litellm` shim (`headroom/backends/litellm.py`). That shim
//! lossy-converts every request and response between Anthropic and
//! OpenAI shapes, dropping `thinking`, `redacted_thinking`,
//! `document`, `search_result`, `image`, `server_tool_use`, and
//! `mcp_tool_use` blocks (P4-37). It also hardcodes
//! `stop_sequence: null` (§11.1 violation) and re-wraps
//! `function_call.arguments` as a parsed JSON object (§4.4 — P4-43).
//!
//! Phase D rebuilds the Bedrock surface natively in Rust. PR-D1
//! handles the **non-streaming** `POST /model/{model}/invoke` route:
//!
//! 1. Parse the Bedrock envelope (`{"anthropic_version": "...",
//!    ...rest_of_anthropic_body}`).
//! 2. Route Anthropic-shape bodies through the live-zone compression
//!    path (the same one `/v1/messages` uses).
//! 3. Re-emit the envelope with `anthropic_version` preserved as the
//!    first key — Bedrock is strict about schema validation.
//! 4. Sign the **outgoing** body bytes with AWS SigV4 (after
//!    compression) and forward to the configured Bedrock endpoint.
//!
//! # Cache safety
//!
//! The signed bytes are exactly the bytes Bedrock receives. If the
//! compressor mutated the body, the SigV4 signature is computed
//! against the post-compression bytes; the upstream verifier will
//! accept them. There is no "sign before compress" path — that would
//! produce a signature that doesn't match the wire payload.
//!
//! # Module layout
//!
//! - [`envelope`] — `BedrockEnvelope` parse + emit (preserves
//!   `anthropic_version` ordering byte-equal).
//! - [`sigv4`] — AWS SigV4 signing helper. Wraps the `aws-sigv4`
//!   crate with the project's no-fallback / structured-logging
//!   policy.
//! - [`invoke`] — POST handler for `/model/{model}/invoke`.
//!
//! Streaming (`/model/{model}/invoke-with-response-stream`) is
//! Phase D PR-D2.

pub mod envelope;
pub mod invoke;
pub mod sigv4;

pub use envelope::{BedrockEnvelope, EnvelopeError};
pub use invoke::handle_invoke;
pub use sigv4::{sign_request, SigV4Error, SigningInputs};
