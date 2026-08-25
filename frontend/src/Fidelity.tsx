import type { Environment, RequirementVerdict } from "./types";

/** What the brief asked for, and what the circuit actually carries.
 *
 *  Deliberately separate from the playability line in `SceneFacts`. That one says the
 *  circuit works; this one says whether it is the circuit that was asked for, and the two
 *  are different questions — a perfectly valid lap can ignore half the brief. The misses
 *  are listed first and quoted back in the user's own words, because "you asked for a
 *  hairpin in the bottom left and did not get one" is information, while a satisfaction
 *  percentage on its own is not.
 */
export function Fidelity({ environment }: { environment: Environment }) {
  const report = environment.fidelity;
  const spec = environment.prompt_spec;
  if (!report || report.verdicts.length === 0) {
    // Older records predate the contract, and an empty panel reads better than a
    // confident "0 requirements" that would look like the brief asked for nothing.
    return spec?.unsupported?.length ? <section className="fidelity">
      <NotSupported items={spec.unsupported} />
    </section> : null;
  }

  const missed = report.verdicts.filter((item) => !item.satisfied);
  const honoured = report.verdicts.filter((item) => item.satisfied);
  const faithful = missed.every((item) => item.priority !== "must");

  return <section className="fidelity">
    <header className="fidelity-head">
      <h2>Prompt fidelity</h2>
      <span className={faithful ? "fidelity-score ok" : "fidelity-score miss"}>
        {honoured.length}/{report.verdicts.length} requirements
      </span>
    </header>

    {missed.length > 0 && <div className="fidelity-group">
      <p className="fidelity-label">Could not be faithful on</p>
      {missed.map((item) => <Verdict key={item.id} verdict={item} />)}
    </div>}

    {honoured.length > 0 && <div className="fidelity-group">
      <p className="fidelity-label">Delivered</p>
      {honoured.map((item) => <Verdict key={item.id} verdict={item} />)}
    </div>}

    {spec?.unsupported?.length ? <NotSupported items={spec.unsupported} /> : null}

    {spec?.unspecified?.length ? <div className="fidelity-group">
      <p className="fidelity-label">You left these open, so the generator chose</p>
      <p className="fidelity-open">{spec.unspecified.join(" · ")}</p>
    </div> : null}
  </section>;
}

function Verdict({ verdict }: { verdict: RequirementVerdict }) {
  return <div className={verdict.satisfied ? "requirement" : "requirement missed"}>
    <span className="requirement-id">{verdict.id}</span>
    <div className="requirement-body">
      <p className="requirement-statement">
        {verdict.statement}
        {verdict.priority === "should" && <em className="requirement-soft"> · soft preference</em>}
      </p>
      {verdict.quote && <p className="requirement-quote">you asked: “{verdict.quote}”</p>}
      {verdict.evidence && <p className="requirement-evidence">
        {/* Named so the reader knows whether a number settled this or an opinion did. */}
        <span className="requirement-method">{verdict.method === "judge" ? "judged" : verdict.method === "unverifiable" ? "unmeasured" : "measured"}</span>
        {verdict.evidence}
      </p>}
    </div>
  </div>;
}

function NotSupported({ items }: { items: string[] }) {
  return <div className="fidelity-group">
    <p className="fidelity-label">The engine cannot do this at all</p>
    {items.map((item) => <p key={item} className="fidelity-unsupported">{item}</p>)}
  </div>;
}
