import { useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";

export interface SelectOption<T extends string = string> {
  value: T;
  label: string;
  disabled?: boolean;
}

interface SelectFieldProps<T extends string = string> {
  ariaLabel?: string;
  className?: string;
  disabled?: boolean;
  options: ReadonlyArray<SelectOption<T>>;
  placeholder?: string;
  value: T;
  onChange: (value: T) => void;
}

export function SelectField<T extends string>({
  ariaLabel,
  className,
  disabled = false,
  options,
  placeholder = "请选择",
  value,
  onChange,
}: SelectFieldProps<T>) {
  const [isOpen, setIsOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const listboxId = useId();
  const selectedOption = options.find((option) => option.value === value);
  const isDisabled = disabled || options.length === 0;

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    function handleMouseDown(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    }

    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  function handleTriggerKeyDown(event: ReactKeyboardEvent<HTMLButtonElement>) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setIsOpen(true);
    }
  }

  function handleSelect(nextValue: T) {
    onChange(nextValue);
    setIsOpen(false);
  }

  return (
    <div className={`select-field ${isOpen ? "is-open" : ""} ${className ?? ""}`.trim()} ref={rootRef}>
      <button
        aria-controls={isOpen ? listboxId : undefined}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
        className="select-field-trigger"
        disabled={isDisabled}
        onClick={() => setIsOpen((current) => !current)}
        onKeyDown={handleTriggerKeyDown}
        type="button"
      >
        <span className={`select-field-value ${selectedOption ? "" : "is-placeholder"}`.trim()}>
          {selectedOption?.label ?? placeholder}
        </span>
        <span aria-hidden="true" className="select-field-arrow" />
      </button>
      {isOpen ? (
        <div className="select-field-menu" id={listboxId} role="listbox">
          {options.map((option) => (
            <button
              aria-selected={option.value === value}
              className={`select-field-option ${option.value === value ? "is-selected" : ""}`.trim()}
              disabled={option.disabled}
              key={option.value}
              onClick={() => handleSelect(option.value)}
              role="option"
              type="button"
            >
              <span>{option.label}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
