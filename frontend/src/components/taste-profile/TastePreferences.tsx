"use client";

import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Plus, RotateCcw, Trash2 } from "lucide-react";
import { toast } from "sonner";

import {
  ApiError,
  createTastePreference,
  deleteTastePreference,
  getTastePreferenceOptions,
  getTastePreferences,
  resetTastePreferences,
  updateTastePreference,
} from "@/lib/api";
import type {
  PreferenceCreateRequest,
  PreferenceCurrency,
  PreferencePatchRequest,
  PreferenceResponse,
  PreferenceStance,
  PreferenceSubjectKind,
  WineStyle,
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

const PREFERENCE_QUERY_KEY = ["taste-preferences"] as const;
const OPTIONS_QUERY_KEY = ["taste-preference-options"] as const;

const SUBJECT_LABELS: Record<PreferenceSubjectKind, string> = {
  grape: "Grapes",
  region: "Regions",
  producer: "Producers",
  wine_style: "Wine styles",
  price_ceiling: "Price ceiling",
};

const STANCE_LABELS: Record<PreferenceStance, string> = {
  like: "Like",
  dislike: "Dislike",
  avoid: "Avoid",
};

type DialogState =
  | { kind: "add" }
  | { kind: "edit"; preference: PreferenceResponse }
  | { kind: "delete"; preference: PreferenceResponse }
  | { kind: "reset" }
  | null;

type MutationAction =
  | { kind: "create"; request: PreferenceCreateRequest }
  | { kind: "update"; id: number; request: PreferencePatchRequest }
  | { kind: "delete"; id: number; version: number }
  | { kind: "reset"; count: number };

function formatError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return "The preference request could not be completed.";
}

function visiblePreferenceValue(preference: PreferenceResponse): string {
  if (preference.subject_kind === "price_ceiling") {
    return preference.price_amount && preference.currency
      ? `${preference.price_amount} ${preference.currency}`
      : "Price ceiling";
  }
  return preference.display_value ?? "Preference value unavailable";
}

function fieldOptions(
  kind: PreferenceSubjectKind,
  options: Awaited<ReturnType<typeof getTastePreferenceOptions>>,
): string[] {
  if (kind === "grape") return options.grapes;
  if (kind === "region") return options.regions;
  if (kind === "producer") return options.producers;
  if (kind === "wine_style") return options.wine_styles;
  return [];
}

function mutationSuccessMessage(kind: MutationAction["kind"]): string {
  if (kind === "create") return "Preference added.";
  if (kind === "update") return "Preference updated.";
  if (kind === "delete") return "Preference deleted.";
  return "Preferences reset.";
}

export default function TastePreferences() {
  const queryClient = useQueryClient();
  const [dialog, setDialog] = useState<DialogState>(null);
  const [dialogError, setDialogError] = useState<string | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  function restoreInitiatorFocus() {
    window.setTimeout(() => returnFocusRef.current?.focus(), 0);
  }

  const preferencesQuery = useQuery({
    queryKey: PREFERENCE_QUERY_KEY,
    queryFn: getTastePreferences,
    staleTime: 0,
    retry: false,
  });
  const optionsQuery = useQuery({
    queryKey: OPTIONS_QUERY_KEY,
    queryFn: getTastePreferenceOptions,
    staleTime: 0,
    retry: false,
  });

  const mutation = useMutation({
    mutationFn: async (action: MutationAction) => {
      if (action.kind === "create") return createTastePreference(action.request);
      if (action.kind === "update") return updateTastePreference(action.id, action.request);
      if (action.kind === "delete") return deleteTastePreference(action.id, action.version);
      return resetTastePreferences({ confirm: true, expected_count: action.count });
    },
    onSuccess: async (_result, action) => {
      await queryClient.invalidateQueries({ queryKey: PREFERENCE_QUERY_KEY });
      toast.success(mutationSuccessMessage(action.kind));
      setDialog(null);
      setDialogError(null);
      restoreInitiatorFocus();
    },
    onError: async (error) => {
      if (error instanceof ApiError && error.status === 409) {
        await queryClient.refetchQueries({ queryKey: PREFERENCE_QUERY_KEY });
        setDialogError(`${error.message} The preference list has been refreshed; review it and try again.`);
        return;
      }
      setDialogError(formatError(error));
    },
  });

  const groups = useMemo(() => {
    const grouped = new Map<PreferenceSubjectKind, PreferenceResponse[]>();
    for (const preference of preferencesQuery.data?.items ?? []) {
      const current = grouped.get(preference.subject_kind) ?? [];
      current.push(preference);
      grouped.set(preference.subject_kind, current);
    }
    return grouped;
  }, [preferencesQuery.data?.items]);

  function openDialog(next: Exclude<DialogState, null>, trigger: HTMLElement) {
    returnFocusRef.current = trigger;
    setDialogError(null);
    mutation.reset();
    setDialog(next);
  }

  function closeDialog() {
    if (mutation.isPending) return;
    setDialog(null);
    setDialogError(null);
    restoreInitiatorFocus();
  }

  if (preferencesQuery.isPending) {
    return (
      <div role="status" aria-live="polite" className="space-y-3" data-testid="preferences-loading">
        <span className="sr-only">Loading preferences</span>
        <div className="h-12 animate-pulse rounded-xl bg-muted" />
        <div className="h-12 animate-pulse rounded-xl bg-muted" />
      </div>
    );
  }

  if (preferencesQuery.isError) {
    return (
      <Card role="alert" className="border-destructive/40">
        <CardHeader>
          <CardTitle>Unable to load preferences</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">{formatError(preferencesQuery.error)}</p>
          <Button type="button" variant="outline" onClick={() => preferencesQuery.refetch()}>
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  const response = preferencesQuery.data;
  const addUnavailable = optionsQuery.isPending || optionsQuery.isError || response.total >= response.max_items;

  return (
    <section className="min-w-0 space-y-6" aria-labelledby="taste-preferences-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 id="taste-preferences-heading" className="font-heading text-2xl font-semibold">
            Your wine preferences
          </h2>
          <p className="mt-1 max-w-3xl text-sm leading-relaxed text-muted-foreground">
            These optional settings guide preference-aware chat recommendations and wine comparisons. They do not
            change your tasting history, Favorites, or the Cellar&apos;s Drink Next list.
          </p>
        </div>
        <Button
          type="button"
          disabled={addUnavailable}
          onClick={(event) => openDialog({ kind: "add" }, event.currentTarget)}
        >
          <Plus aria-hidden="true" />
          Add preference
        </Button>
      </div>

      {optionsQuery.isError && (
        <div role="alert" className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm">
          Existing preferences remain available, but Add is disabled because form options could not be loaded.
          <Button
            type="button"
            variant="link"
            className="ml-1 h-auto p-0"
            onClick={() => optionsQuery.refetch()}
          >
            Retry options
          </Button>
        </div>
      )}

      <div className="grid overflow-hidden rounded-xl border border-border bg-card lg:grid-cols-[1.3fr_repeat(3,1fr)]">
        <div className="bg-brand-burgundy/5 p-4">
          <h3 className="font-semibold">How these settings work</h3>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
            Choose grapes, regions, producers, styles, or one price ceiling. You stay in control; nothing is inferred
            or added automatically.
          </p>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            Price ceilings use recorded bottle prices in the same currency; wines with missing or different-currency
            prices may not qualify.
          </p>
        </div>
        <div className="border-t border-border p-4 lg:border-l lg:border-t-0">
          <h3 className="font-semibold text-green-700 dark:text-green-400">Like</h3>
          <p className="mt-1 text-sm text-muted-foreground">Raises the preference score when a wine matches.</p>
        </div>
        <div className="border-t border-border p-4 lg:border-l lg:border-t-0">
          <h3 className="font-semibold">Dislike</h3>
          <p className="mt-1 text-sm text-muted-foreground">Lowers the score, but does not block a recommendation.</p>
        </div>
        <div className="border-t border-border p-4 lg:border-l lg:border-t-0">
          <h3 className="font-semibold text-destructive">Avoid</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Excludes matching wines from profile-based recommendations.
          </p>
        </div>
      </div>

      {response.total === 0 ? (
        <Card className="border-dashed bg-card">
          <CardContent className="py-10 text-center">
            <h3 className="font-heading text-lg font-medium">No wine preferences yet</h3>
            <p className="mx-auto mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
              Setup is optional. Add a grape, region, producer, wine style, or price ceiling when you want
              preference-aware recommendations to reflect your current intent.
            </p>
            <Button
              type="button"
              className="mt-5"
              disabled={addUnavailable}
              onClick={(event) => openDialog({ kind: "add" }, event.currentTarget)}
            >
              <Plus aria-hidden="true" />
              Add your first preference
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid items-start gap-4 md:grid-cols-2 xl:grid-cols-3">
          {[...groups.entries()].map(([kind, preferences]) => (
            <section
              key={kind}
              aria-labelledby={`preference-group-${kind}`}
              className="min-w-0 overflow-hidden rounded-xl border border-border bg-card"
            >
              <div className="border-b border-border bg-muted/30 px-4 py-2.5">
                <h3 id={`preference-group-${kind}`} className="font-heading text-sm font-semibold">
                  {SUBJECT_LABELS[kind]}
                </h3>
              </div>
              <ul aria-label={`${SUBJECT_LABELS[kind]} preferences`} className="divide-y divide-border">
                {preferences.map((preference) => (
                  <li key={preference.id} className="flex min-w-0 items-center gap-3 px-3 py-2.5 sm:px-4">
                    <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
                      <p className="min-w-0 break-words font-medium">{visiblePreferenceValue(preference)}</p>
                      <div className="flex shrink-0 flex-wrap items-center gap-1.5">
                        {preference.stance && <Badge variant="secondary">{STANCE_LABELS[preference.stance]}</Badge>}
                        {preference.subject_kind === "price_ceiling" && <Badge variant="secondary">Ceiling</Badge>}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        aria-label={`Edit ${visiblePreferenceValue(preference)}`}
                        title={`Edit ${visiblePreferenceValue(preference)}`}
                        onClick={(event) => openDialog({ kind: "edit", preference }, event.currentTarget)}
                      >
                        <Pencil aria-hidden="true" />
                      </Button>
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        aria-label={`Delete ${visiblePreferenceValue(preference)}`}
                        title={`Delete ${visiblePreferenceValue(preference)}`}
                        onClick={(event) => openDialog({ kind: "delete", preference }, event.currentTarget)}
                      >
                        <Trash2 aria-hidden="true" className="text-destructive" />
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}

      {response.total > 0 && (
        <div className="border-t border-border pt-5">
          <Card>
            <CardContent className="flex flex-wrap items-center justify-between gap-3 p-4">
              <div>
                <h3 className="font-heading font-medium">Preference management</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Reset removes only these explicit settings. It does not change ratings or tasting history.
                </p>
              </div>
              <Button
                type="button"
                variant="outline"
                className="text-destructive hover:text-destructive"
                onClick={(event) => openDialog({ kind: "reset" }, event.currentTarget)}
              >
                <RotateCcw aria-hidden="true" />
                Reset all preferences
              </Button>
            </CardContent>
          </Card>
        </div>
      )}

      <Dialog open={dialog !== null} onOpenChange={(open) => !open && closeDialog()}>
        {dialog?.kind === "add" && optionsQuery.data && (
          <AddPreferenceDialog
            options={optionsQuery.data}
            error={dialogError}
            pending={mutation.isPending}
            onCancel={closeDialog}
            onSubmit={(request) => mutation.mutate({ kind: "create", request })}
          />
        )}
        {dialog?.kind === "edit" && (
          <EditPreferenceDialog
            preference={dialog.preference}
            options={optionsQuery.data}
            error={dialogError}
            pending={mutation.isPending}
            onCancel={closeDialog}
            onSubmit={(request) => mutation.mutate({ kind: "update", id: dialog.preference.id, request })}
          />
        )}
        {dialog?.kind === "delete" && (
          <DeletePreferenceDialog
            preference={dialog.preference}
            error={dialogError}
            pending={mutation.isPending}
            onCancel={closeDialog}
            onConfirm={() =>
              mutation.mutate({ kind: "delete", id: dialog.preference.id, version: dialog.preference.version })
            }
          />
        )}
        {dialog?.kind === "reset" && (
          <ResetPreferencesDialog
            count={response.total}
            error={dialogError}
            pending={mutation.isPending}
            onCancel={closeDialog}
            onConfirm={() => mutation.mutate({ kind: "reset", count: response.total })}
          />
        )}
      </Dialog>
    </section>
  );
}

interface DialogCommonProps {
  error: string | null;
  pending: boolean;
  onCancel: () => void;
}

function DialogStatus({ error, pending }: Pick<DialogCommonProps, "error" | "pending">) {
  return (
    <div aria-live="polite" aria-atomic="true" className="min-h-5 text-sm">
      {pending && <span className="text-muted-foreground">Saving preference…</span>}
      {error && <span id="preference-dialog-error" className="text-destructive">{error}</span>}
    </div>
  );
}

function FormField({
  id,
  label,
  error,
  children,
}: {
  id: string;
  label: string;
  error?: string | null;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="text-sm font-medium">{label}</label>
      {children}
      {error && <p id={`${id}-error`} className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

interface AddDialogProps extends DialogCommonProps {
  options: Awaited<ReturnType<typeof getTastePreferenceOptions>>;
  onSubmit: (request: PreferenceCreateRequest) => void;
}

function AddPreferenceDialog({ options, error, pending, onCancel, onSubmit }: AddDialogProps) {
  const initialKind = options.subject_kinds[0] ?? "grape";
  const [kind, setKind] = useState<PreferenceSubjectKind>(initialKind);
  const [stance, setStance] = useState<PreferenceStance>(options.stances[0] ?? "like");
  const [value, setValue] = useState(() => fieldOptions(initialKind, options)[0] ?? "");
  const [priceAmount, setPriceAmount] = useState("");
  const [currency, setCurrency] = useState<PreferenceCurrency>(options.currencies[0] ?? "EUR");
  const [validationError, setValidationError] = useState<string | null>(null);
  const availableValues = fieldOptions(kind, options);

  function changeKind(next: PreferenceSubjectKind) {
    setKind(next);
    setValue(fieldOptions(next, options)[0] ?? "");
    setValidationError(null);
  }

  function submit(event: React.FormEvent) {
    event.preventDefault();
    if (kind === "price_ceiling") {
      if (!/^(?:0|[1-9][0-9]{0,5})(?:\.[0-9]{1,2})?$/.test(priceAmount) || Number(priceAmount) < 0.01) {
        setValidationError("Enter an amount from 0.01 to 999999.99 with at most two decimal places.");
        return;
      }
      onSubmit({ subject_kind: kind, price_amount: priceAmount, currency });
      return;
    }
    if (!value) {
      setValidationError("Select a preference value.");
      return;
    }
    if (kind === "wine_style") {
      onSubmit({ subject_kind: kind, stance, value: value as WineStyle });
      return;
    }
    onSubmit({ subject_kind: kind, stance, value });
  }

  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Add preference</DialogTitle>
        <DialogDescription>
          Add an explicit preference used by future recommendations and comparisons.
        </DialogDescription>
      </DialogHeader>
      <form onSubmit={submit} className="space-y-4">
        <FormField id="preference-kind" label="Preference type">
          <select
            id="preference-kind"
            value={kind}
            disabled={pending}
            onChange={(event) => changeKind(event.target.value as PreferenceSubjectKind)}
            className="h-9 w-full rounded-lg border border-input bg-background px-2.5 text-sm"
          >
            {options.subject_kinds.map((subjectKind) => (
              <option key={subjectKind} value={subjectKind}>{SUBJECT_LABELS[subjectKind]}</option>
            ))}
          </select>
        </FormField>

        {kind === "price_ceiling" ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <FormField id="preference-price" label="Maximum price" error={validationError}>
              <Input
                id="preference-price"
                inputMode="decimal"
                value={priceAmount}
                disabled={pending}
                aria-invalid={Boolean(validationError)}
                aria-describedby={validationError ? "preference-price-error" : undefined}
                onChange={(event) => setPriceAmount(event.target.value)}
              />
            </FormField>
            <FormField id="preference-currency" label="Currency">
              <select
                id="preference-currency"
                value={currency}
                disabled={pending}
                onChange={(event) => setCurrency(event.target.value as PreferenceCurrency)}
                className="h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm"
              >
                {options.currencies.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </FormField>
          </div>
        ) : (
          <>
            <FormField id="preference-stance" label="Stance">
              <select
                id="preference-stance"
                value={stance}
                disabled={pending}
                onChange={(event) => setStance(event.target.value as PreferenceStance)}
                className="h-9 w-full rounded-lg border border-input bg-background px-2.5 text-sm"
              >
                {options.stances.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </FormField>
            <FormField id="preference-value" label="Value" error={validationError}>
              <select
                id="preference-value"
                value={value}
                disabled={pending || availableValues.length === 0}
                aria-invalid={Boolean(validationError)}
                aria-describedby={validationError ? "preference-value-error" : undefined}
                onChange={(event) => setValue(event.target.value)}
                className="h-9 w-full rounded-lg border border-input bg-background px-2.5 text-sm"
              >
                {availableValues.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </FormField>
          </>
        )}
        <DialogStatus error={error} pending={pending} />
        <DialogFooter>
          <Button type="button" variant="outline" disabled={pending} onClick={onCancel}>Cancel</Button>
          <Button type="submit" disabled={pending}>Add preference</Button>
        </DialogFooter>
      </form>
    </DialogContent>
  );
}

interface EditDialogProps extends DialogCommonProps {
  preference: PreferenceResponse;
  options?: Awaited<ReturnType<typeof getTastePreferenceOptions>>;
  onSubmit: (request: PreferencePatchRequest) => void;
}

function EditPreferenceDialog({ preference, options, error, pending, onCancel, onSubmit }: EditDialogProps) {
  const [stance, setStance] = useState<PreferenceStance>(preference.stance ?? "like");
  const [priceAmount, setPriceAmount] = useState(preference.price_amount ?? "");
  const [currency, setCurrency] = useState<PreferenceCurrency>(preference.currency ?? "EUR");
  const [validationError, setValidationError] = useState<string | null>(null);

  function submit(event: React.FormEvent) {
    event.preventDefault();
    if (preference.subject_kind === "price_ceiling") {
      if (!/^(?:0|[1-9][0-9]{0,5})(?:\.[0-9]{1,2})?$/.test(priceAmount) || Number(priceAmount) < 0.01) {
        setValidationError("Enter an amount from 0.01 to 999999.99 with at most two decimal places.");
        return;
      }
      onSubmit({ expected_version: preference.version, price_amount: priceAmount, currency });
      return;
    }
    onSubmit({ expected_version: preference.version, stance });
  }

  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Edit {visiblePreferenceValue(preference)}</DialogTitle>
        <DialogDescription>
          Only mutable preference details can be changed.
        </DialogDescription>
      </DialogHeader>
      <form onSubmit={submit} className="space-y-4">
        {preference.subject_kind === "price_ceiling" ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <FormField id="edit-preference-price" label="Maximum price" error={validationError}>
              <Input
                id="edit-preference-price"
                inputMode="decimal"
                value={priceAmount}
                disabled={pending}
                aria-invalid={Boolean(validationError)}
                aria-describedby={validationError ? "edit-preference-price-error" : undefined}
                onChange={(event) => setPriceAmount(event.target.value)}
              />
            </FormField>
            <FormField id="edit-preference-currency" label="Currency">
              <select
                id="edit-preference-currency"
                value={currency}
                disabled={pending}
                onChange={(event) => setCurrency(event.target.value as PreferenceCurrency)}
                className="h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm"
              >
                {(options?.currencies ?? [currency]).map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </FormField>
          </div>
        ) : (
          <FormField id="edit-preference-stance" label="Stance">
            <select
              id="edit-preference-stance"
              value={stance}
              disabled={pending}
              onChange={(event) => setStance(event.target.value as PreferenceStance)}
              className="h-9 w-full rounded-lg border border-input bg-background px-2.5 text-sm"
            >
              {(options?.stances ?? [stance]).map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </FormField>
        )}
        <DialogStatus error={error} pending={pending} />
        <DialogFooter>
          <Button type="button" variant="outline" disabled={pending} onClick={onCancel}>Cancel</Button>
          <Button type="submit" disabled={pending}>Save changes</Button>
        </DialogFooter>
      </form>
    </DialogContent>
  );
}

interface DeleteDialogProps extends DialogCommonProps {
  preference: PreferenceResponse;
  onConfirm: () => void;
}

function DeletePreferenceDialog({ preference, error, pending, onCancel, onConfirm }: DeleteDialogProps) {
  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Delete preference</DialogTitle>
        <DialogDescription>
          Delete {visiblePreferenceValue(preference)}? This immediately removes it from future recommendations.
        </DialogDescription>
      </DialogHeader>
      <DialogStatus error={error} pending={pending} />
      <DialogFooter>
        <Button type="button" variant="outline" disabled={pending} onClick={onCancel}>Cancel</Button>
        <Button type="button" variant="destructive" disabled={pending} onClick={onConfirm}>
          Delete preference
        </Button>
      </DialogFooter>
    </DialogContent>
  );
}

interface ResetDialogProps extends DialogCommonProps {
  count: number;
  onConfirm: () => void;
}

function ResetPreferencesDialog({ count, error, pending, onCancel, onConfirm }: ResetDialogProps) {
  const [confirmation, setConfirmation] = useState("");

  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Reset all preferences</DialogTitle>
        <DialogDescription>
          Permanently delete all {count} declared {count === 1 ? "preference" : "preferences"}. Type RESET to confirm.
        </DialogDescription>
      </DialogHeader>
      <FormField id="reset-preferences-confirmation" label="Type RESET">
        <Input
          id="reset-preferences-confirmation"
          value={confirmation}
          disabled={pending}
          autoComplete="off"
          onChange={(event) => setConfirmation(event.target.value)}
        />
      </FormField>
      <DialogStatus error={error} pending={pending} />
      <DialogFooter>
        <Button type="button" variant="outline" disabled={pending} onClick={onCancel}>Cancel</Button>
        <Button
          type="button"
          variant="destructive"
          disabled={pending || confirmation !== "RESET"}
          onClick={onConfirm}
        >
          Reset all preferences
        </Button>
      </DialogFooter>
    </DialogContent>
  );
}
