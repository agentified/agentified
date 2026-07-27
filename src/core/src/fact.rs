use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// Whether a [`Fact`] is *always* injected or only surfaced *when a query
/// retrieves it* — the one bit that splits the always-on tier from the
/// retrieval-gated tier.
///
/// The core does not inject anything itself; `pin` is metadata the higher
/// layers (the SDK grounding path) act on. [`crate::FactRegistry::pinned`]
/// filters the corpus by it, and both variants are ranked by
/// [`crate::FactRegistry::search`] all the same, so a pinned fact is still
/// discoverable by a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PinMode {
    /// Always injected into the context, no ranking — the barbershop's address,
    /// hours, brand voice. Kept tiny; it is paid for on every injection.
    Always,
    /// Injected only when a query ranks it in — pricing tables, per-service
    /// policies. The default: a new fact is retrieval-gated until promoted.
    #[default]
    Retrieved,
}

impl PinMode {
    /// The wire/`as_str` identifier used across the SDKs: `"always"` or
    /// `"retrieved"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            PinMode::Always => "always",
            PinMode::Retrieved => "retrieved",
        }
    }
}

impl fmt::Display for PinMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The identifier did not name a known pin mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsePinModeError(pub String);

impl fmt::Display for ParsePinModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown pin mode {:?} (expected \"always\" or \"retrieved\")",
            self.0
        )
    }
}

impl std::error::Error for ParsePinModeError {}

impl FromStr for PinMode {
    type Err = ParsePinModeError;

    /// Parse the SDK identifier: `"always"` or `"retrieved"`.
    ///
    /// # Errors
    ///
    /// Any other string is a [`ParsePinModeError`] naming the rejected input.
    ///
    /// # Examples
    ///
    /// ```
    /// use ratel_ai_core::PinMode;
    ///
    /// assert_eq!("always".parse::<PinMode>(), Ok(PinMode::Always));
    /// assert_eq!("retrieved".parse::<PinMode>(), Ok(PinMode::Retrieved));
    /// assert!("pinned".parse::<PinMode>().is_err());
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "always" => Ok(PinMode::Always),
            "retrieved" => Ok(PinMode::Retrieved),
            other => Err(ParsePinModeError(other.to_string())),
        }
    }
}

/// A fact registered for grounding — constant, declarative context an agent
/// needs to have on hand (a barbershop's address and hours, a brand's voice).
/// The push-path analog of a [`crate::Skill`]: where a skill is a playbook the
/// agent *pulls* and runs on demand, a fact is content the grounding layer
/// *pushes* into the context so the model is never missing it.
///
/// `name`, `description`, and `tags` drive ranking (they are folded into the
/// searchable text) exactly as on a [`crate::Skill`], so the retrieval-gated
/// tier is discoverable by query. `body` is the injected content — **not**
/// indexed, so a long body never skews relevance. `pin` splits the tiers (see
/// [`PinMode`]); `metadata` is free-form, non-indexed context for higher-layer
/// biasing, never matched as query terms.
pub struct Fact {
    /// Stable identifier, returned in [`crate::FactHit::fact_id`] and stamped
    /// on the grounding trace events. Registering the same id again replaces
    /// the entry in place. Not indexed for ranking.
    pub id: String,
    /// Short name. Indexed both verbatim and identifier-split, so
    /// snake_case/camelCase/kebab constituent words match.
    pub name: String,
    /// What the fact is about — the primary ranking text (not the content
    /// itself; that is `body`).
    pub description: String,
    /// Author-declared labels and task phrases, indexed alongside the
    /// description.
    pub tags: Vec<String>,
    /// Free-form, non-indexed context for higher layers (push-path
    /// boosting/filtering); never matched as query terms.
    pub metadata: HashMap<String, Vec<String>>,
    /// The fact's content — the payload injected into the context, not indexed
    /// (a long body would otherwise drown the description's term weights).
    pub body: String,
    /// Always-on vs retrieval-gated (see [`PinMode`]). Not indexed.
    pub pin: PinMode,
}
