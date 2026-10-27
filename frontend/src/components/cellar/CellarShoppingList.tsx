"use client";

import { FormEvent, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ExternalLink, Pencil, Plus, ShoppingCart, Trash2 } from "lucide-react";
import { toast } from "sonner";

import {
  ApiError,
  createShoppingListItem,
  deleteShoppingListItem,
  getShoppingList,
  getShoppingListCatalogueOptions,
  updateShoppingListItem,
} from "@/lib/api";
import type {
  ShoppingListCatalogueOption,
  ShoppingListCreateRequest,
  ShoppingListItem,
  ShoppingListPatchRequest,
  ShoppingListPriority,
  ShoppingListStatus,
} from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

const PAGE_SIZE = 25;
const SHOPPING_LIST_QUERY_KEY = ["shopping-list"] as const;
const fieldClassName =
  "h-8 w-full rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30";

type DialogState =
  | { kind: "add" }
  | { kind: "edit"; item: ShoppingListItem }
  | { kind: "purchase"; item: ShoppingListItem }
  | { kind: "delete"; item: ShoppingListItem }
  | null;

type MutationAction =
  | { kind: "create"; request: ShoppingListCreateRequest; duplicateSearch: string }
  | { kind: "update"; id: number; request: ShoppingListPatchRequest }
  | { kind: "purchase"; item: ShoppingListItem }
  | { kind: "restore"; item: ShoppingListItem }
  | { kind: "delete"; item: ShoppingListItem };

function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : "The shopping-list request could not be completed.";
}

function displayName(item: Pick<ShoppingListItem, "name" | "vintage">): string {
  return item.vintage ? `${item.name} ${item.vintage}` : item.name;
}

function safeRetailerUrl(value: string | null): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? url.toString() : null;
  } catch {
    return null;
  }
}

function mutationSuccessMessage(kind: MutationAction["kind"]): string {
  if (kind === "create") return "Wine added to your shopping list.";
  if (kind === "update") return "Shopping-list item updated.";
  if (kind === "purchase") return "Item marked purchased. Cellar inventory was not changed.";
  if (kind === "restore") return "Item moved back to Wanted.";
  return "Shopping-list item deleted.";
}

export default function CellarShoppingList() {
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<ShoppingListStatus>("wanted");
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);
  const [dialog, setDialog] = useState<DialogState>(null);
  const [dialogError, setDialogError] = useState<string | null>(null);
  const [listActionError, setListActionError] = useState<string | null>(null);
  const [duplicateTarget, setDuplicateTarget] = useState<{ id: number; search: string } | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  const listQuery = useQuery({
    queryKey: [...SHOPPING_LIST_QUERY_KEY, status, search, offset],
    queryFn: () => getShoppingList({ status, search: search.trim() || undefined, limit: PAGE_SIZE, offset }),
    retry: false,
  });

  function closeDialog() {
    if (mutation.isPending) return;
    setDialog(null);
    setDialogError(null);
    setDuplicateTarget(null);
    window.setTimeout(() => returnFocusRef.current?.focus(), 0);
  }

  function openDialog(next: Exclude<DialogState, null>, trigger: HTMLElement) {
    returnFocusRef.current = trigger;
    setDialogError(null);
    setDuplicateTarget(null);
    mutation.reset();
    setDialog(next);
  }

  const mutation = useMutation({
    mutationFn: async (action: MutationAction) => {
      if (action.kind === "create") return createShoppingListItem(action.request);
      if (action.kind === "update") return updateShoppingListItem(action.id, action.request);
      if (action.kind === "purchase") {
        return updateShoppingListItem(action.item.id, {
          expected_version: action.item.version,
          status: "purchased",
        });
      }
      if (action.kind === "restore") {
        return updateShoppingListItem(action.item.id, {
          expected_version: action.item.version,
          status: "wanted",
        });
      }
      return deleteShoppingListItem(action.item.id, action.item.version);
    },
    onSuccess: async (_result, action) => {
      await queryClient.invalidateQueries({ queryKey: SHOPPING_LIST_QUERY_KEY });
      toast.success(mutationSuccessMessage(action.kind));
      setDialog(null);
      setDialogError(null);
      setListActionError(null);
      setDuplicateTarget(null);
      window.setTimeout(() => returnFocusRef.current?.focus(), 0);
    },
    onError: async (error, action) => {
      if (error instanceof ApiError && error.status === 409) {
        await queryClient.refetchQueries({ queryKey: SHOPPING_LIST_QUERY_KEY });
        if (action.kind === "create" && error.details?.existing_item_id) {
          setDuplicateTarget({ id: error.details.existing_item_id, search: action.duplicateSearch });
          setDialogError(`${error.message} Your current form values have been kept.`);
          return;
        }
        const message = `${error.message} The list has been refreshed; review the latest item and try again.`;
        if (action.kind === "restore") setListActionError(message);
        else setDialogError(message);
        return;
      }
      const message = `${errorMessage(error)} Your current form values have been kept.`;
      if (action.kind === "restore") setListActionError(message);
      else setDialogError(message);
    },
  });

  function showExistingDuplicate() {
    if (!duplicateTarget) return;
    setStatus("wanted");
    setSearch(duplicateTarget.search);
    setOffset(0);
    setDialog(null);
    setDialogError(null);
    setDuplicateTarget(null);
  }

  const response = listQuery.data;
  const canGoBack = offset > 0;
  const canGoForward = response ? offset + response.items.length < response.total : false;

  return (
    <section className="min-w-0 space-y-4" aria-labelledby="shopping-list-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="shopping-list-heading" className="font-heading text-xl font-semibold">Shopping List</h2>
          <p className="mt-1 text-sm text-muted-foreground">Plan future purchases without changing cellar inventory.</p>
        </div>
        <Button type="button" onClick={(event) => openDialog({ kind: "add" }, event.currentTarget)}>
          <Plus aria-hidden="true" /> Add wine
        </Button>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="inline-flex w-fit rounded-lg border border-border p-1" aria-label="Shopping-list status">
          {(["wanted", "purchased"] as const).map((nextStatus) => (
            <Button
              key={nextStatus}
              type="button"
              size="sm"
              variant={status === nextStatus ? "default" : "ghost"}
              aria-pressed={status === nextStatus}
              onClick={() => { setStatus(nextStatus); setOffset(0); }}
            >
              {nextStatus === "wanted" ? "Wanted" : "Purchased"}
            </Button>
          ))}
        </div>
        <label className="w-full sm:max-w-xs">
          <span className="sr-only">Search shopping list</span>
          <Input
            type="search"
            value={search}
            placeholder="Search by wine name"
            onChange={(event) => { setSearch(event.target.value); setOffset(0); }}
          />
        </label>
      </div>

      {listQuery.isPending && (
        <div role="status" aria-live="polite" className="space-y-3">
          <span className="sr-only">Loading shopping list</span>
          <div className="h-28 animate-pulse rounded-xl bg-muted" />
          <div className="h-28 animate-pulse rounded-xl bg-muted" />
        </div>
      )}

      {listQuery.isError && (
        <Card role="alert" className="border-destructive/40">
          <CardHeader><CardTitle>Unable to load the shopping list</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">{errorMessage(listQuery.error)}</p>
            <Button type="button" variant="outline" onClick={() => listQuery.refetch()}>Retry</Button>
          </CardContent>
        </Card>
      )}

      {listActionError && (
        <div role="alert" className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm">
          {listActionError}
        </div>
      )}

      {response && response.items.length === 0 && (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            {search ? "No wines match this search." : status === "wanted" ? "Your shopping list is empty." : "No purchased items yet."}
          </CardContent>
        </Card>
      )}

      {response && response.items.length > 0 && (
        <div className="space-y-3" aria-live="polite">
          {response.items.map((item) => (
            <ShoppingListRow
              key={item.id}
              item={item}
              busy={mutation.isPending}
              onEdit={(trigger) => openDialog({ kind: "edit", item }, trigger)}
              onPurchase={(trigger) => openDialog({ kind: "purchase", item }, trigger)}
              onRestore={(trigger) => {
                returnFocusRef.current = trigger;
                setDialogError(null);
                setListActionError(null);
                mutation.mutate({ kind: "restore", item });
              }}
              onDelete={(trigger) => openDialog({ kind: "delete", item }, trigger)}
            />
          ))}
        </div>
      )}

      {response && response.total > 0 && (
        <div className="flex items-center justify-between gap-3 text-sm text-muted-foreground">
          <span>Showing {offset + 1}–{Math.min(offset + response.items.length, response.total)} of {response.total}</span>
          <div className="flex gap-2">
            <Button type="button" variant="outline" disabled={!canGoBack} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>Previous</Button>
            <Button type="button" variant="outline" disabled={!canGoForward} onClick={() => setOffset(offset + PAGE_SIZE)}>Next</Button>
          </div>
        </div>
      )}

      <Dialog open={dialog?.kind === "add" || dialog?.kind === "edit"} onOpenChange={(open) => { if (!open) closeDialog(); }}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>{dialog?.kind === "edit" ? "Edit shopping-list item" : "Add wine to shopping list"}</DialogTitle>
            <DialogDescription>Shopping-list changes do not add or remove cellar inventory.</DialogDescription>
          </DialogHeader>
          {(dialog?.kind === "add" || dialog?.kind === "edit") && (
            <ShoppingListForm
              key={dialog.kind === "edit" ? `edit-${dialog.item.id}-${dialog.item.version}` : "add"}
              item={dialog.kind === "edit" ? dialog.item : undefined}
              pending={mutation.isPending}
              error={dialogError}
              duplicateTarget={duplicateTarget}
              onShowDuplicate={showExistingDuplicate}
              onCancel={closeDialog}
              onSubmit={(request, duplicateSearch) => {
                if (dialog.kind === "edit") {
                  mutation.mutate({ kind: "update", id: dialog.item.id, request: { expected_version: dialog.item.version, ...request } });
                } else {
                  mutation.mutate({ kind: "create", request: request as ShoppingListCreateRequest, duplicateSearch });
                }
              }}
            />
          )}
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={dialog?.kind === "purchase"}
        title="Mark this item purchased?"
        description="This records the shopping-list status only. It will not add bottles to cellar inventory."
        confirmLabel="Mark purchased"
        pending={mutation.isPending}
        error={dialogError}
        onClose={closeDialog}
        onConfirm={() => { if (dialog?.kind === "purchase") mutation.mutate({ kind: "purchase", item: dialog.item }); }}
      />
      <ConfirmDialog
        open={dialog?.kind === "delete"}
        title={dialog?.kind === "delete" ? `Delete ${displayName(dialog.item)}?` : "Delete item?"}
        description="This removes the item from the shopping list. This action cannot be undone."
        confirmLabel="Delete item"
        destructive
        pending={mutation.isPending}
        error={dialogError}
        onClose={closeDialog}
        onConfirm={() => { if (dialog?.kind === "delete") mutation.mutate({ kind: "delete", item: dialog.item }); }}
      />
    </section>
  );
}

function ShoppingListRow({
  item,
  busy,
  onEdit,
  onPurchase,
  onRestore,
  onDelete,
}: {
  item: ShoppingListItem;
  busy: boolean;
  onEdit: (trigger: HTMLElement) => void;
  onPurchase: (trigger: HTMLElement) => void;
  onRestore: (trigger: HTMLElement) => void;
  onDelete: (trigger: HTMLElement) => void;
}) {
  const retailerUrl = safeRetailerUrl(item.retailer_url);
  return (
    <Card className="py-3">
      <CardContent className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto] md:items-center">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="font-heading font-semibold">{displayName(item)}</h3>
            {item.priority === "high" && <Badge variant="destructive">High priority</Badge>}
            <Badge variant="secondary">Qty {item.quantity}</Badge>
            {item.identity_kind === "manual" && <Badge variant="outline">Manual</Badge>}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted-foreground">
            {item.link_available && item.wine_id && <a className="underline underline-offset-4 hover:text-foreground" href={`/cellar/${item.wine_id}`}>View wine</a>}
            {retailerUrl && <a className="inline-flex items-center gap-1 underline underline-offset-4 hover:text-foreground" href={retailerUrl} target="_blank" rel="noopener noreferrer">Retailer <ExternalLink className="size-3" aria-hidden="true" /></a>}
          </div>
          {item.notes && <details className="text-sm"><summary className="cursor-pointer text-muted-foreground">Notes</summary><p className="mt-1 whitespace-pre-wrap">{item.notes}</p></details>}
        </div>
        <div className="flex flex-wrap gap-2 md:justify-end">
          <Button type="button" variant="outline" disabled={busy} onClick={(event) => onEdit(event.currentTarget)}><Pencil aria-hidden="true" /> Edit</Button>
          {item.status === "wanted" ? (
            <Button type="button" disabled={busy} onClick={(event) => onPurchase(event.currentTarget)}><ShoppingCart aria-hidden="true" /> Mark purchased</Button>
          ) : (
            <Button type="button" variant="outline" disabled={busy} onClick={(event) => onRestore(event.currentTarget)}>Move to Wanted</Button>
          )}
          <Button type="button" variant="destructive" disabled={busy} aria-label={`Delete ${displayName(item)}`} onClick={(event) => onDelete(event.currentTarget)}><Trash2 aria-hidden="true" /></Button>
        </div>
      </CardContent>
    </Card>
  );
}

function ShoppingListForm({
  item,
  pending,
  error,
  duplicateTarget,
  onShowDuplicate,
  onCancel,
  onSubmit,
}: {
  item?: ShoppingListItem;
  pending: boolean;
  error: string | null;
  duplicateTarget: { id: number; search: string } | null;
  onShowDuplicate: () => void;
  onCancel: () => void;
  onSubmit: (request: ShoppingListCreateRequest | Omit<ShoppingListPatchRequest, "expected_version">, duplicateSearch: string) => void;
}) {
  const [mode, setMode] = useState<"catalogue" | "manual">(item?.identity_kind ?? "catalogue");
  const [catalogueInput, setCatalogueInput] = useState("");
  const [catalogueSearch, setCatalogueSearch] = useState("");
  const [catalogueOffset, setCatalogueOffset] = useState(0);
  const [selectedOption, setSelectedOption] = useState<ShoppingListCatalogueOption | null>(null);
  const [name, setName] = useState(item?.name ?? "");
  const [vintage, setVintage] = useState(item?.vintage?.toString() ?? "");
  const [quantity, setQuantity] = useState(item?.quantity.toString() ?? "1");
  const [priority, setPriority] = useState<ShoppingListPriority>(item?.priority ?? "normal");
  const [notes, setNotes] = useState(item?.notes ?? "");
  const [retailerUrl, setRetailerUrl] = useState(item?.retailer_url ?? "");
  const [formError, setFormError] = useState<string | null>(null);

  const catalogueQuery = useQuery({
    queryKey: ["shopping-list-catalogue", catalogueSearch, catalogueOffset],
    queryFn: () => getShoppingListCatalogueOptions(catalogueSearch, catalogueOffset),
    enabled: !item && mode === "catalogue" && catalogueSearch.length > 0,
    retry: false,
  });

  function submitCatalogueSearch(event: FormEvent) {
    event.preventDefault();
    const value = catalogueInput.trim();
    if (!value) { setFormError("Enter a wine name before searching the catalogue."); return; }
    setFormError(null);
    setSelectedOption(null);
    setCatalogueOffset(0);
    setCatalogueSearch(value);
  }

  function useDifferentVintage() {
    if (!selectedOption) return;
    setName(selectedOption.name);
    setVintage(selectedOption.vintage?.toString() ?? "");
    setMode("manual");
    setFormError(null);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    const parsedQuantity = Number(quantity);
    const parsedVintage = vintage ? Number(vintage) : null;
    if (!Number.isInteger(parsedQuantity) || parsedQuantity < 1 || parsedQuantity > 999) {
      setFormError("Quantity must be a whole number from 1 to 999."); return;
    }
    if (vintage && (!Number.isInteger(parsedVintage) || (parsedVintage ?? 0) < 1000 || (parsedVintage ?? 0) > 9999)) {
      setFormError("Vintage must be a four-digit year."); return;
    }
    const common = { quantity: parsedQuantity, priority, notes: notes.trim() || null, retailer_url: retailerUrl.trim() || null };
    if (item) {
      onSubmit({ ...common, ...(item.identity_kind === "manual" ? { name: name.trim(), vintage: parsedVintage } : {}) }, item.name);
      return;
    }
    if (mode === "catalogue") {
      if (!selectedOption) { setFormError("Choose one catalogue wine before adding it."); return; }
      onSubmit({ identity_kind: "catalogue", wine_id: selectedOption.id, ...common }, selectedOption.name);
      return;
    }
    if (!name.trim()) { setFormError("Wine name is required."); return; }
    onSubmit({ identity_kind: "manual", name: name.trim(), vintage: parsedVintage, ...common }, name.trim());
  }

  const visibleError = formError ?? error;
  return (
    <div className="space-y-4">
      {!item && (
        <div className="grid grid-cols-2 gap-2" aria-label="Wine identity mode">
          <Button type="button" variant={mode === "catalogue" ? "default" : "outline"} aria-pressed={mode === "catalogue"} onClick={() => { setMode("catalogue"); setFormError(null); }}>From catalogue</Button>
          <Button type="button" variant={mode === "manual" ? "default" : "outline"} aria-pressed={mode === "manual"} onClick={() => { setMode("manual"); setFormError(null); }}>Enter manually</Button>
        </div>
      )}

      {!item && mode === "catalogue" && (
        <div className="space-y-3">
          <form className="flex gap-2" onSubmit={submitCatalogueSearch}>
            <label className="min-w-0 flex-1"><span className="sr-only">Catalogue wine name</span><Input value={catalogueInput} onChange={(event) => setCatalogueInput(event.target.value)} placeholder="Search local catalogue" /></label>
            <Button type="submit" variant="outline">Search</Button>
          </form>
          {catalogueQuery.isPending && catalogueSearch && <p role="status" className="text-sm text-muted-foreground">Searching catalogue…</p>}
          {catalogueQuery.isError && <p role="alert" className="text-sm text-destructive">{errorMessage(catalogueQuery.error)}</p>}
          {catalogueQuery.data && catalogueQuery.data.items.length === 0 && <p className="text-sm text-muted-foreground">No catalogue wines matched. Try another search or enter the wine manually.</p>}
          {catalogueQuery.data && catalogueQuery.data.items.length > 0 && (
            <fieldset className="space-y-2"><legend className="text-sm font-medium">Choose a catalogue wine</legend>
              {catalogueQuery.data.items.map((option) => (
                <label key={option.id} className="flex cursor-pointer items-start gap-3 rounded-lg border border-border p-3 has-checked:border-brand-burgundy has-checked:bg-muted/50">
                  <input type="radio" name="catalogue-option" className="mt-1" checked={selectedOption?.id === option.id} onChange={() => setSelectedOption(option)} />
                  <span className="text-sm"><span className="font-medium">{option.name}{option.vintage ? ` ${option.vintage}` : ""}</span><span className="block text-muted-foreground">{[option.producer, option.bottle_size].filter(Boolean).join(" · ")}</span></span>
                </label>
              ))}
              <div className="flex justify-end gap-2">
                {selectedOption && (
                  <Button type="button" variant="outline" onClick={useDifferentVintage}>
                    Use a different vintage
                  </Button>
                )}
                <Button type="button" variant="outline" disabled={catalogueOffset === 0} onClick={() => setCatalogueOffset(Math.max(0, catalogueOffset - 20))}>Previous matches</Button>
                <Button type="button" variant="outline" disabled={!catalogueQuery.data.has_more} onClick={() => setCatalogueOffset(catalogueOffset + 20)}>More matches</Button>
              </div>
            </fieldset>
          )}
        </div>
      )}

      <form className="space-y-4" onSubmit={submit}>
        {(mode === "manual" || item?.identity_kind === "manual") && (
          <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_8rem]">
            <label className="space-y-1"><span className="text-sm font-medium">Wine name</span><Input required maxLength={300} value={name} onChange={(event) => setName(event.target.value)} /></label>
            <label className="space-y-1"><span className="text-sm font-medium">Vintage</span><Input type="number" min={1000} max={9999} value={vintage} onChange={(event) => setVintage(event.target.value)} /></label>
          </div>
        )}
        {item?.identity_kind === "catalogue" && <p className="rounded-lg bg-muted p-3 text-sm"><span className="font-medium">{displayName(item)}</span><span className="block text-muted-foreground">Catalogue identity cannot be changed.</span></p>}
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1"><span className="text-sm font-medium">Quantity</span><Input required type="number" min={1} max={999} value={quantity} onChange={(event) => setQuantity(event.target.value)} /></label>
          <label className="space-y-1"><span className="text-sm font-medium">Priority</span><select className={fieldClassName} value={priority} onChange={(event) => setPriority(event.target.value as ShoppingListPriority)}><option value="normal">Normal</option><option value="high">High</option></select></label>
        </div>
        <label className="block space-y-1"><span className="text-sm font-medium">Retailer URL</span><Input type="url" maxLength={2048} value={retailerUrl} onChange={(event) => setRetailerUrl(event.target.value)} placeholder="https://…" /></label>
        <label className="block space-y-1"><span className="text-sm font-medium">Notes</span><textarea className={`${fieldClassName} min-h-24 py-2`} maxLength={2000} value={notes} onChange={(event) => setNotes(event.target.value)} /></label>
        {visibleError && <div role="alert" className="space-y-2 rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm"><p>{visibleError}</p>{duplicateTarget && <Button type="button" variant="outline" onClick={onShowDuplicate}>Show existing item #{duplicateTarget.id}</Button>}</div>}
        <DialogFooter>
          <Button type="button" variant="outline" disabled={pending} onClick={onCancel}>Cancel</Button>
          <Button type="submit" disabled={pending}>{pending ? "Saving…" : item ? "Save changes" : "Add to list"}</Button>
        </DialogFooter>
      </form>
    </div>
  );
}

function ConfirmDialog({ open, title, description, confirmLabel, destructive = false, pending, error, onClose, onConfirm }: { open: boolean; title: string; description: string; confirmLabel: string; destructive?: boolean; pending: boolean; error: string | null; onClose: () => void; onConfirm: () => void }) {
  return (
    <Dialog open={open} onOpenChange={(nextOpen) => { if (!nextOpen) onClose(); }}>
      <DialogContent>
        <DialogHeader><DialogTitle>{title}</DialogTitle><DialogDescription>{description}</DialogDescription></DialogHeader>
        {error && <p role="alert" className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm">{error}</p>}
        <DialogFooter>
          <Button type="button" variant="outline" disabled={pending} onClick={onClose}>Cancel</Button>
          <Button type="button" variant={destructive ? "destructive" : "default"} disabled={pending} onClick={onConfirm}>{pending ? "Saving…" : confirmLabel}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
