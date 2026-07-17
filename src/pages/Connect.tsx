import { useState } from "react";
import { Link } from "react-router-dom";
import { Copy, Check, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const mcpUrl = `https://${import.meta.env.VITE_SUPABASE_PROJECT_ID}.supabase.co/functions/v1/mcp`;

export default function Connect() {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(mcpUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="min-h-screen bg-background">
      <main className="max-w-3xl mx-auto px-6 py-12">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-6">
          <ArrowLeft className="h-4 w-4" /> Back
        </Link>

        <h1 className="text-3xl font-semibold tracking-tight mb-2">Connect an AI assistant</h1>
        <p className="text-muted-foreground mb-8">
          Give ChatGPT or Claude access to this app's tools for validating, summarizing, and comparing ML result JSON files.
        </p>

        <Card className="p-5 mb-10">
          <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">MCP server URL</div>
          <div className="flex items-center gap-2">
            <code className="flex-1 px-3 py-2 rounded bg-muted text-sm font-mono break-all">{mcpUrl}</code>
            <Button onClick={copy} variant="outline" size="sm" className="shrink-0">
              {copied ? <><Check className="h-4 w-4 mr-1" /> Copied</> : <><Copy className="h-4 w-4 mr-1" /> Copy</>}
            </Button>
          </div>
        </Card>

        <section className="mb-10">
          <h2 className="text-xl font-semibold mb-4">Connect</h2>

          <div className="mb-6">
            <h3 className="font-medium mb-2">ChatGPT</h3>
            <ol className="list-decimal pl-5 space-y-1 text-sm text-muted-foreground">
              <li>Open <a className="text-primary hover:underline" href="https://chatgpt.com/#settings/Connectors/Advanced" target="_blank" rel="noreferrer">ChatGPT Connectors → Advanced</a> and enable Developer mode (read the risk notice shown there).</li>
              <li>In the chat composer's "+" menu, turn on Developer mode.</li>
              <li>Click "Add sources", then "Connect more".</li>
              <li>Name the connector and paste the MCP URL above.</li>
              <li>Ask ChatGPT to use the app.</li>
            </ol>
          </div>

          <div>
            <h3 className="font-medium mb-2">Claude</h3>
            <ol className="list-decimal pl-5 space-y-1 text-sm text-muted-foreground">
              <li>Open <a className="text-primary hover:underline" href="https://claude.ai/customize/connectors?modal=add-custom-connector" target="_blank" rel="noreferrer">Claude custom connectors</a>.</li>
              <li>Name the connector and paste the MCP URL above.</li>
              <li>Enable the connector from the chat composer, then ask Claude to use the app.</li>
            </ol>
          </div>
        </section>

        <section className="mb-10">
          <h2 className="text-xl font-semibold mb-4">Refresh after the app changes</h2>
          <p className="text-sm text-muted-foreground mb-4">
            Connected assistants cache the tool list. After we ship updates, refresh the connector to pick them up.
          </p>

          <div className="mb-6">
            <h3 className="font-medium mb-2">ChatGPT</h3>
            <ol className="list-decimal pl-5 space-y-1 text-sm text-muted-foreground">
              <li>Open ChatGPT's app preferences and pick this app under "Enabled apps".</li>
              <li>Next to "Information", click "Refresh".</li>
              <li>If the URL changed, paste the latest URL from above.</li>
              <li>Start a new chat and ask ChatGPT to use the app.</li>
            </ol>
          </div>

          <div>
            <h3 className="font-medium mb-2">Claude</h3>
            <ol className="list-decimal pl-5 space-y-1 text-sm text-muted-foreground">
              <li>Open the Connectors page and select this connector.</li>
              <li>Refresh or update the connector's tools.</li>
              <li>If the URL changed, paste the latest URL from above.</li>
              <li>Ask Claude to use the app.</li>
            </ol>
          </div>
        </section>

        <footer className="mt-16 pt-8 border-t border-border text-center text-sm text-muted-foreground">
          <p>
            Multi-Method ML Classifier • Powered by{" "}
            <a href="https://accelbio.pt/" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline font-medium">AccelBio</a>
          </p>
        </footer>
      </main>
    </div>
  );
}