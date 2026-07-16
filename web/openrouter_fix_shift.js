import { app } from "../../scripts/app.js";

// Кнопка "Fix api_key shift" для OpenRouterNode.
// Старые воркфлоу держат устаревший api_key в widgets_values[0]; лишнее значение
// сдвигает все виджеты на 1 (ComfyUI мапит widgets_values позиционно) — визуально
// поля съезжают (ключ в system_prompt и т.д.). Бэкенд-шим чинит это на исполнении,
// а эта кнопка чистит ВИЗУАЛЬНО в редакторе: сдвигает значения назад и убирает ключ.
// Ключ нам не нужен — реальный берётся из ENV.

function looksLikeStaleKey(v) {
    if (typeof v !== "string") return false;
    const s = v.trim();
    return s.startsWith("sk-or-") || s.toUpperCase().includes("OPENROUTER_API_KEY");
}

// Сдвигает значения value-виджетов на 1 назад, если в первом застрял api_key.
// Возвращает true, если что-то починил. Кнопки (type "button") исключаются.
function fixShiftedNode(node) {
    const vw = (node.widgets || []).filter((w) => w && w.type !== "button");
    if (vw.length < 2) return false;
    if (!looksLikeStaleKey(vw[0].value)) return false;
    for (let i = 0; i < vw.length - 1; i++) {
        vw[i].value = vw[i + 1].value;
    }
    // Последнее значение потеряно при сдвиге -> ставим дефолт виджета, если он есть.
    const last = vw[vw.length - 1];
    if (last.options && Object.prototype.hasOwnProperty.call(last.options, "default")) {
        last.value = last.options.default;
    }
    return true;
}

app.registerExtension({
    name: "Desow.OpenRouter.FixApiKeyShift",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== "OpenRouterNode") return;
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = orig ? orig.apply(this, arguments) : undefined;
            const btn = this.addWidget("button", "🧹 Fix api_key shift", null, () => {
                let fixed = 0;
                const nodes = (app.graph && app.graph._nodes) || [];
                for (const n of nodes) {
                    if (((n.comfyClass || n.type) === "OpenRouterNode") && fixShiftedNode(n)) {
                        fixed++;
                    }
                }
                if (app.graph) app.graph.setDirtyCanvas(true, true);
                alert(
                    fixed
                        ? "Почищено нод: " + fixed + ". Сохраните воркфлоу, чтобы зафиксировать."
                        : "Сдвиг не обнаружен — чистить нечего."
                );
            });
            // КРИТИЧНО: кнопка не должна попадать в widgets_values (иначе сама сдвинет поля).
            if (btn) btn.serialize = false;
            return r;
        };
    },
});
